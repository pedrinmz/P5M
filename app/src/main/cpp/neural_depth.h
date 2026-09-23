// SPDX-License-Identifier: AGPL-3.0-only
//
// Profundidade por rede neural, rodando em paralelo ao frame loop.
//
// Modelo usado: MiDaS v2.1 small, 256x256, fp16 -- o mesmo arquivo que o
// projeto moonlight-android-xr ja usa em producao (13.5ms por inferencia na
// GPU de um Quest, contra 183-265ms na CPU, nos testes deles). Entrada RGB
// float em 0..1 (a normalizacao ImageNet ja esta dentro do grafo, nao e
// preciso fazer fora); saida um unico canal, profundidade relativa inversa
// em escala arbitraria -- maior e mais perto, mas o valor bruto nao vem em
// nenhuma unidade fixa, precisa ser renormalizado quadro a quadro (ver
// PollResult, mais abaixo).
//
// ## Por que uma thread propria
//
// O frame loop do OpenXR tem um orcamento fixo -- a 90Hz sao 11ms por quadro,
// e o compositor descarta o que nao chega a tempo. Uma rede de profundidade,
// mesmo pequena, nao cabe nesse orcamento com folga suficiente para nao virar
// o gargalo: se ela rodasse inline, um quadro lento da rede vira um quadro
// perdido do XR inteiro, e isso e o tipo de engasgo que da desconforto em VR.
//
// A saida e desacoplar: a rede roda na cadencia que ela consegue -- pode ser
// 30, pode ser 45 Hz, nao precisa ser 90 -- e o frame loop sempre le o
// resultado mais recente pronto, nunca espera. Profundidade nao muda tao
// rapido quanto a imagem: uma estimativa com um ou dois quadros de atraso e
// impercetivel, ao contrario de video atrasado.
//
// ## O caminho de dados, hoje (versao simples)
//
// 1. Na thread do frame loop, apos RenderDepth() heuristico normal: a fonte
//    (textura externa OES) ja esta ligada; um blit pequeno (FBO de
//    kInputW x kInputH) copia para uma textura RGBA8 comum -- o mesmo tipo de
//    downsample barato que a passada de profundidade heuristica ja faz, so
//    que num alvo ainda menor, do tamanho da entrada da rede.
// 2. glReadPixels desse alvo pequeno (poucos KB) para um buffer CPU. Este
//    buffer fica em convencao GL: linha 0 e a linha de BAIXO da imagem, nao
//    a de cima -- e assim que glReadPixels sempre devolve, nao e escolha
//    nossa. Ver o comentario de SubmitFrame().
// 3. O buffer e entregue a thread de inferencia por um par de buffers em
//    rodizio protegido por mutex -- so o mais recente importa, quadros
//    perdidos entre duas inferencias sao descartados de proposito.
// 4. A thread de inferencia roda o interpretador TFLite (delegate GPU, com
//    CPU como reserva) e escreve o mapa de profundidade resultante num
//    buffer CPU proprio, ja renormalizado para [0, 1] e com a linha
//    desvirada de volta pra convencao GL (ver PollResult()).
// 5. No inicio do proximo quadro, a thread do frame loop pergunta se ha
//    resultado novo; se houver, faz um glTexSubImage2D pequeno na textura de
//    ping-pong que alimenta ToneMapper::SetNeuralDepthTexture.
//
// Isto tem duas copias CPU<->GPU que uma versao "zero-copy" eliminaria --
// AHardwareBuffer compartilhado entre o produtor GL e o delegate GPU do
// TFLite, sem passar pela CPU. Vale a pena depois de medir: a copia de uma
// imagem de 256x256 e da ordem de 260KB, e em muitos aparelhos isso e ruido
// perto do custo da propria inferencia. Comece por aqui, meça no Quest 3 e só
// vá atrás do caminho sem cópia se o profiler apontar a copia como o gargalo.
//
// ## O que falta para isto compilar de verdade
//
// Este arquivo assume a TensorFlow Lite C API (ou LiteRT, o sucessor dela).
// Ela não vem com o projeto -- é preciso:
//   1. Adicionar a dependência no build.gradle do app, por exemplo:
//        implementation("org.tensorflow:tensorflow-lite:2.16.1")
//        implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
//      (isto baixa da internet; não há como fazer isso neste ambiente sem
//      rede -- precisa rodar no seu próprio Android Studio / Gradle.)
//   2. Expor os .so e headers ao CMake via prefab (o AAR do TFLite já publica
//      prefab desde a 2.9), e linkar tensorflowlite_c e, se for usar GPU,
//      tensorflowlite_gpu_delegate no CMakeLists.txt.
//   3. O modelo já está em app/src/main/assets/midas_v21_small_256_fp16.tflite
//      -- copiado do moonlight-android-xr, nenhuma conversão própria
//      necessária. Ver docs/3D-NEURAL-INTEGRACAO.md para a proveniência.
// Até isso ser feito, P5M_WITH_NEURAL_DEPTH fica indefinido, os métodos desta
// classe viram no-ops que devolvem "sem resultado", e o 3D continua
// funcionando exatamente como hoje, na estimativa heurística.
#pragma once

#include <GLES3/gl3.h>
#include <cstdint>

#if __has_include(<tensorflow/lite/c/c_api.h>)
#define P5M_WITH_NEURAL_DEPTH 1
#endif

namespace p5m {

class NeuralDepth
{
public:
	// 256x256, quadrado -- nao e uma escolha nossa, e o que o MiDaS v2.1 small
	// exige. Ao contrario de uma rede totalmente convolucional que aceita
	// qualquer tamanho multiplo de 32, esta foi exportada para uma entrada
	// fixa: o .tflite ja vem "travado" em 256x256, sem eixo dinamico. Um
	// video 16:9 fica com uma leve distorcao horizontal ao ser encaixado num
	// quadrado -- imperceptivel no mapa de profundidade, mesmo raciocinio de
	// antes.
	static constexpr int32_t kInputW = 256;
	static constexpr int32_t kInputH = 256;

	~NeuralDepth() { Destroy(); }

	/**
	 * Carrega o modelo dos assets e sobe a thread de inferencia.
	 *
	 * `asset_manager` vem de AAssetManager_fromJava, no lado Java/Kotlin.
	 * `model_asset_path` e o caminho dentro de assets/, por exemplo
	 * "midas_v21_small_256_fp16.tflite" -- o modelo que ja vem com o projeto.
	 *
	 * Devolve false se o TFLite nao estiver linkado (P5M_WITH_NEURAL_DEPTH
	 * indefinido) ou se o modelo nao carregar -- nos dois casos o chamador
	 * deve simplesmente nao usar esta classe, e o 3D cai para a estimativa
	 * heuristica sem mais nada a fazer.
	 */
	bool Init(void *asset_manager, const char *model_asset_path);

	/**
	 * Entrega um quadro novo para a fila de inferencia.
	 *
	 * `rgba` e `kInputW * kInputH * 4` bytes, RGBA8, ja no tamanho da rede --
	 * quem chama fez o downsample (ver o comentario de topo, passo 1-2). As
	 * linhas vem em convencao GL -- a linha 0 de `rgba` e a linha de BAIXO da
	 * imagem, exatamente como glReadPixels devolve. Nao inverta antes de
	 * chamar: a inversao para a orientacao que a rede espera (a "normal",
	 * chao embaixo) acontece dentro desta classe, porque e so aqui que se
	 * sabe o que a rede precisa -- o chamador nao precisa saber disso.
	 *
	 * Nao bloqueia: se a thread de inferencia ainda estiver ocupada com o
	 * quadro anterior, este e o que fica esperando e um quadro do meio pode
	 * ser descartado. E a decisao certa aqui -- profundidade atrasada de um
	 * quadro nao incomoda, profundidade que trava o produtor incomoda.
	 */
	void SubmitFrame(const uint8_t *rgba);

	/**
	 * Se ha um resultado novo desde a ultima chamada, escreve nele e devolve
	 * true. `out_depth` precisa caber `kInputW * kInputH` floats.
	 *
	 * Ja vem pronto para subir direto numa textura GL_R8: renormalizado para
	 * [0, 1] (a rede devolve profundidade relativa em escala arbitraria, nao
	 * em [0,1] -- a normalizacao e por quadro, com o minimo/maximo suavizados
	 * ao longo do tempo para o mapa nao piscar quando a cena corta) e com a
	 * linha de volta na convencao GL, simetrico ao que SubmitFrame recebeu.
	 *
	 * Chamar da thread do frame loop, uma vez por quadro, antes de decidir se
	 * chama ToneMapper::SetNeuralDepthTexture. Sem resultado novo devolve
	 * false e nao mexe em `out_depth` -- o chamador mantem o quadro anterior
	 * (ou a estimativa heuristica, se ainda nao houve nenhum).
	 */
	bool PollResult(float *out_depth);

	void Destroy();

	bool ready() const { return ready_; }

private:
	struct Impl;
	Impl *impl_ = nullptr;
	bool ready_ = false;
};

} // namespace p5m

