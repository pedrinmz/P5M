# 3D por profundidade neural — o que já está pronto e o que falta

## O que muda

O 3D sintetizado do P5M vinha inteiramente de um shader heurístico
(`kDepthShader` em `tone_mapper.cpp`): bordas, saturação e uma pista de "chão"
combinadas por pesos fixos. Funciona, mas erra em cenas sem chão visível,
levanta texto/HUD e não generaliza para tipos de cena diferentes.

Esta integração adiciona um segundo caminho — uma rede neural de profundidade
monocular — sem remover o heurístico: os dois convivem, e o heurístico é o
fallback automático sempre que a rede não está pronta ou falha.

## O modelo

**MiDaS v2.1 small**, 256×256, exportado em fp16 —
`app/src/main/assets/midas_v21_small_256_fp16.tflite` (33MB). Não foi
convertido por nós: é o mesmo arquivo que o projeto `moonlight-android-xr`
já usa em produção, com o script de conversão deles preservado em
`docs/convert_midas_referencia.py` para o dia em que for preciso gerar outra
variante (INT8, outra resolução, outro modelo). Os números de latência deles
no Quest: **13.5ms por inferência com o delegate de GPU, contra 183–265ms na
CPU** — a diferença é grande o bastante para o delegate de GPU não ser
opcional na prática.

Contrato do modelo (o que `neural_depth.cpp` assume, e checa na inicialização
comparando com o tensor real do `.tflite`):
- **Entrada**: `1×256×256×3`, float32, RGB em `[0, 1]` — sem normalização
  ImageNet por fora, ela já está dentro do grafo.
- **Saída**: `1×256×256×1`, float32, profundidade relativa *inversa* em
  escala arbitrária — maior número é mais perto, mas não há unidade fixa;
  precisa ser renormalizada por quadro (feito em `neural_depth.cpp`, com a
  faixa min/max suavizada ao longo do tempo para não piscar em corte de
  cena).
- **Convenção de linha**: a imagem que sai do `glReadPixels` do lado GL vem
  com a linha 0 sendo a linha de *baixo* (convenção do OpenGL). A rede se
  importa com qual lado é chão, então `neural_depth.cpp` inverte a imagem
  antes de entregar pra rede, e desinverte o resultado antes de devolver —
  as duas inversões acontecem dentro da classe; o resto do pipeline nunca
  vê nem precisa saber disso.

## Arquitetura, em uma frase

Uma thread própria roda a inferência em paralelo ao frame loop do OpenXR;
todo quadro novo, o `ToneMapper` baixa a imagem da fonte para 256×256, entrega
para essa thread, e sobe para a GPU o resultado mais recente que estiver
pronto — sem nunca esperar pela rede dentro do frame loop.

```
MediaCodec --textura externa OES--> ToneMapper::Render()
                                         |
                                         v
                          UpdateNeuralDepth() [a cada quadro novo]
                                         |
                    downsample (GL) -> glReadPixels -> SubmitFrame()
                                         |
                         (thread de inferência: inverte linha,
                          RGBA8->RGB float 0..1, TfLiteInterpreterInvoke,
                          normaliza saída com faixa suavizada, desinverte)
                                         |
                                    PollResult()
                                         |
                             glTexSubImage2D -> neural_result_tex_
                                         |
                                         v
                    kDepthShader (uNeuralDepth, uUseNeural=1)
                                         |
                    [interface_ mask + EMA temporal, já existentes]
                                         |
                                    depth_tex_ (igual a antes)
                                         |
                                         v
                    kFragmentShader (parallax por olho, sem mudança)
```

## Arquivos

- `neural_depth.h` / `neural_depth.cpp` — módulo completo. Thread de
  inferência, fila de um lugar só (mais recente sempre vence), TFLite C API
  com delegate de GPU, conversão de entrada/saída (inversão de linha,
  normalização por faixa suavizada) já implementadas — não há mais `TODO`
  pendente aqui. **Compila e funciona como no-op sem a dependência do
  TFLite** — ver `P5M_WITH_NEURAL_DEPTH` no topo do header.
- `tone_mapper.h` / `tone_mapper.cpp` — `SetNeuralDepth()`,
  `SetNeuralDepthTexture()`, `EnsureNeuralTargets()`, `UpdateNeuralDepth()`,
  o shader `kNeuralDownsampleShader`, e o branch `uUseNeural` dentro de
  `kDepthShader`.
- `xr_session.h` / `xr_session.cpp` — dono do `NeuralDepth` (`neural_depth_`),
  inicializado em `Create()` a partir do `AAssetManager` da Activity, ligado
  ao `ToneMapper` com `SetNeuralDepth(&neural_depth_)`.
- `CMakeLists.txt` — `neural_depth.cpp` na lista de fontes; instruções
  comentadas para linkar o TFLite quando a dependência existir.
- `app/src/main/assets/midas_v21_small_256_fp16.tflite` — o modelo, pronto.
- `docs/convert_midas_referencia.py` — script de conversão original (do
  moonlight-android-xr), preservado como referência, não precisa ser rodado.

## O que falta para isto rodar de verdade

Bem mais curto agora que o modelo já está convertido e a dependência já está
no `app/build.gradle` e no `CMakeLists.txt`:

1. **Clonar o repositório de verdade com `--recursive`** (ver
   `docs/COMO-COMPILAR-3D-NEURAL.md`) — o zip de onde este projeto veio não
   traz o submódulo `chiaki-ng`, e o `CMakeLists.txt` recusa configurar sem
   ele.

2. **Compilar e instalar no Quest 3.** Nesse ponto o 3D neural já deveria
   funcionar de ponta a ponta — não há mais nenhum `TODO` de código pendente,
   e a dependência do TFLite já está resolvida via Gradle/prefab.

3. **Medir.** FPS, temperatura do aparelho numa sessão longa, e se o
   delegate de GPU do TFLite é suficiente ou se vale migrar para o delegate
   QNN (Hexagon) da Qualcomm — melhor eficiência energética, mais trabalho de
   setup. Se o `glReadPixels` da etapa de downsample aparecer como custo
   relevante no profiler, esse é o próximo ponto a otimizar (caminho
   `AHardwareBuffer` sem cópia CPU↔GPU — não implementado aqui de propósito,
   ver o comentário de topo em `neural_depth.h`; vale notar que o próprio
   moonlight-android-xr, numa implementação já testada em produção, também
   optou pelo `glReadPixels` simples em vez do caminho sem cópia).

4. **Comparar com o heurístico.** Vale literalmente alternar entre os dois
   (é só chamar ou não `tone_mapper_.SetNeuralDepth(&neural_depth_)`) e jogar
   a mesma cena nos dois modos, porque a rede erra de um jeito diferente do
   heurístico — pode ficar pior em alguns tipos de cena (interface pesada,
   por exemplo) mesmo sendo mais correta em geral. Vale ter os dois
   disponíveis como opção de configuração, não só trocar um pelo outro.

## O que não muda

Tudo que já existia continua igual: a máscara de interface/HUD, a suavização
temporal (EMA) sobre o mapa final, o teto de disparidade pela IPD, o modo
estéreo empacotado. A rede neural só substitui a fonte do valor `d` dentro do
`kDepthShader` — o resto do pipeline nem sabe que a fonte mudou.
