// SPDX-License-Identifier: AGPL-3.0-only
#include "neural_depth.h"
#include "log.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if P5M_WITH_NEURAL_DEPTH
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <android/asset_manager.h>
#endif

namespace p5m {

struct NeuralDepth::Impl
{
	std::thread worker;
	std::atomic<bool> running{false};

	// Fila de um lugar so: o produtor (frame loop) escreve o quadro mais
	// recente, o consumidor (thread de inferencia) le o que tiver quando
	// terminar o anterior. Sem fila de verdade de proposito -- profundidade
	// atrasada de mais de um quadro nao serve para nada.
	std::mutex frame_mutex;
	std::condition_variable frame_cv;
	std::vector<uint8_t> pending_frame;
	bool has_pending_frame = false;

	// Resultado: mesma logica, so lugar para o mais recente.
	std::mutex result_mutex;
	std::vector<float> result;
	bool has_new_result = false;

	// Buffer de entrada em float, RGB, 0..1 -- o layout que o tensor do MiDaS
	// espera. Reaproveitado a cada quadro para nao alocar de novo.
	std::vector<float> input_float;

	// Faixa (minimo/maximo) do quadro anterior, suavizada -- ver o comentario
	// em cima do laco da thread de inferencia, mais abaixo, sobre por que a
	// normalizacao nao pode ser so quadro a quadro isolado.
	float smooth_lo = 0.0f;
	float smooth_hi = 1.0f;
	bool range_valid = false;
	// Quanto o novo minimo/maximo pesa a cada quadro -- baixo o bastante para
	// nao piscar num corte de cena, alto o bastante para acompanhar quando a
	// cena muda de verdade em vez de ficar presa numa faixa antiga.
	static constexpr float kRangeAlpha = 0.15f;

#if P5M_WITH_NEURAL_DEPTH
	TfLiteModel *model = nullptr;
	TfLiteInterpreterOptions *options = nullptr;
	TfLiteInterpreter *interpreter = nullptr;
	TfLiteDelegate *gpu_delegate = nullptr;
#endif
};

bool NeuralDepth::Init(void *asset_manager, const char *model_asset_path)
{
#if !P5M_WITH_NEURAL_DEPTH
	(void)asset_manager;
	(void)model_asset_path;
	LOGI("NeuralDepth: TFLite nao linkado neste build; 3D fica na estimativa "
			"heuristica. Ver o comentario de topo em neural_depth.h.");
	return false;
#else
	// O modelo mora em assets/, entao precisa ser lido pelo AAssetManager em
	// vez de um caminho de arquivo comum -- assets dentro do APK nao sao
	// arquivos do sistema.
	auto *mgr = reinterpret_cast<AAssetManager *>(asset_manager);
	AAsset *asset = AAssetManager_open(mgr, model_asset_path, AASSET_MODE_BUFFER);
	if(!asset)
	{
		LOGE("NeuralDepth: modelo '%s' nao encontrado em assets/", model_asset_path);
		return false;
	}
	const off_t size = AAsset_getLength(asset);
	const void *data = AAsset_getBuffer(asset);
	if(!data || size <= 0)
	{
		LOGE("NeuralDepth: leitura de '%s' falhou", model_asset_path);
		AAsset_close(asset);
		return false;
	}

	impl_ = new Impl();
	// TfLiteModelCreate copia o que precisa antes de devolver; o asset pode
	// fechar em seguida.
	impl_->model = TfLiteModelCreate(data, (size_t)size);
	AAsset_close(asset);
	if(!impl_->model)
	{
		LOGE("NeuralDepth: TfLiteModelCreate falhou para '%s'", model_asset_path);
		delete impl_;
		impl_ = nullptr;
		return false;
	}

	impl_->options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(impl_->options, 2);

	// Delegate de GPU: melhor latencia na maioria dos aparelhos Android para
	// um modelo deste tamanho, mas nem todo driver aceita todas as operacoes
	// do grafo -- se a criacao falhar, cai para CPU (NNAPI ou o kernel
	// interpretado) em vez de abortar. Um app de VR sem 3D neural ainda
	// funciona; um app que nao abre nao serve para nada.
	TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
	gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
	gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
	impl_->gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
	if(impl_->gpu_delegate)
		TfLiteInterpreterOptionsAddDelegate(impl_->options, impl_->gpu_delegate);
	else
		LOGI("NeuralDepth: delegate de GPU indisponivel, caindo para CPU");

	impl_->interpreter = TfLiteInterpreterCreate(impl_->model, impl_->options);
	if(!impl_->interpreter || TfLiteInterpreterAllocateTensors(impl_->interpreter) != kTfLiteOk)
	{
		LOGE("NeuralDepth: falha ao criar/alocar o interpretador");
		Destroy();
		return false;
	}

	// A forma real do tensor e a fonte da verdade -- se o .tflite que estiver
	// em assets/ nao for o esperado (outro modelo, outra resolucao), isto
	// pega o descompasso aqui, num log claro, em vez de um crash obscuro ou
	// (pior) um TfLiteTensorCopyFromBuffer que silenciosamente le/escreve do
	// tamanho errado.
	const TfLiteTensor *in_tensor = TfLiteInterpreterGetInputTensor(impl_->interpreter, 0);
	const TfLiteTensor *out_tensor_check = TfLiteInterpreterGetOutputTensor(impl_->interpreter, 0);
	const size_t expected_in = (size_t)kInputW * kInputH * 3 * sizeof(float);
	const size_t expected_out = (size_t)kInputW * kInputH * sizeof(float);
	if(!in_tensor || !out_tensor_check
			|| TfLiteTensorByteSize(in_tensor) != expected_in
			|| TfLiteTensorByteSize(out_tensor_check) != expected_out)
	{
		LOGE("NeuralDepth: forma do tensor nao bate com %dx%d esperado "
				"(entrada %zu bytes, esperado %zu; saida %zu, esperado %zu) -- "
				"'%s' e o modelo certo?", kInputW, kInputH,
				in_tensor ? TfLiteTensorByteSize(in_tensor) : 0, expected_in,
				out_tensor_check ? TfLiteTensorByteSize(out_tensor_check) : 0, expected_out,
				model_asset_path);
		Destroy();
		return false;
	}

	impl_->input_float.resize((size_t)kInputW * kInputH * 3);
	impl_->result.resize((size_t)kInputW * kInputH, 0.5f);
	impl_->running = true;
	impl_->worker = std::thread([this]()
	{
		std::vector<uint8_t> local_frame((size_t)kInputW * kInputH * 4);
		while(impl_->running.load())
		{
			{
				std::unique_lock<std::mutex> lock(impl_->frame_mutex);
				impl_->frame_cv.wait(lock, [this]()
				{
					return impl_->has_pending_frame || !impl_->running.load();
				});
				if(!impl_->running.load())
					break;
				local_frame.swap(impl_->pending_frame);
				impl_->has_pending_frame = false;
			}

			// Conversao de entrada: RGBA8 (GL, linha 0 = linha de baixo da
			// imagem) para RGB float 0..1 (MiDaS, linha 0 = linha de cima --
			// "chao embaixo" importa para uma rede de profundidade do jeito
			// que nao importa para nada geometrico no resto do pipeline GL).
			// A inversao de linha e o descarte do alfa acontecem juntos aqui.
			const int n = kInputW; // quadrado: largura == altura
			for(int y = 0; y < n; y++)
			{
				const uint8_t *src = local_frame.data() + (size_t)(n - 1 - y) * n * 4;
				float *dst = impl_->input_float.data() + (size_t)y * n * 3;
				for(int x = 0; x < n; x++)
				{
					dst[x * 3 + 0] = src[x * 4 + 0] * (1.0f / 255.0f);
					dst[x * 3 + 1] = src[x * 4 + 1] * (1.0f / 255.0f);
					dst[x * 3 + 2] = src[x * 4 + 2] * (1.0f / 255.0f);
				}
			}
			TfLiteTensor *in_tensor = TfLiteInterpreterGetInputTensor(impl_->interpreter, 0);
			if(TfLiteTensorCopyFromBuffer(in_tensor, impl_->input_float.data(),
					impl_->input_float.size() * sizeof(float)) != kTfLiteOk)
			{
				LOGE("NeuralDepth: TfLiteTensorCopyFromBuffer falhou");
				continue;
			}

			if(TfLiteInterpreterInvoke(impl_->interpreter) != kTfLiteOk)
			{
				LOGE("NeuralDepth: TfLiteInterpreterInvoke falhou");
				continue;
			}

			{
				std::lock_guard<std::mutex> lock(impl_->result_mutex);

				const TfLiteTensor *out_tensor = TfLiteInterpreterGetOutputTensor(
						impl_->interpreter, 0);
				// Reaproveita input_float como area de trabalho para a saida
				// crua -- ja nao e mais preciso depois do Invoke() acima, e
				// tem exatamente o tamanho certo (kInputW*kInputH floats,
				// contando so um canal em vez de tres, entao sobra espaco).
				float *raw = impl_->input_float.data();
				if(TfLiteTensorCopyToBuffer(out_tensor, raw, (size_t)n * n * sizeof(float)) != kTfLiteOk)
				{
					LOGE("NeuralDepth: TfLiteTensorCopyToBuffer falhou");
					continue;
				}

				// A rede devolve profundidade relativa inversa em escala
				// arbitraria -- nao ha "1.0 = um metro" nenhum aqui, so
				// "maior numero = mais perto". Pra virar algo que da pra
				// desenhar, a faixa (minimo e maximo do quadro) precisa ser
				// encontrada e remapeada para [0, 1] -- e essa faixa muda de
				// quadro a quadro. Normalizar cada quadro isolado do zero
				// faz a faixa saltar toda vez que a cena corta (de um close
				// escuro pra uma paisagem aberta, por exemplo), e esse salto
				// pisca no mapa de profundidade inteiro de uma vez. Por isso
				// a faixa em si passa por uma media movel (smooth_lo/hi)
				// antes de virar a escala usada no quadro -- a normalizacao
				// muda devagar mesmo quando o conteudo muda rápido.
				float lo = raw[0], hi = raw[0];
				for(int i = 1; i < n * n; i++)
				{
					if(raw[i] < lo) lo = raw[i];
					if(raw[i] > hi) hi = raw[i];
				}
				if(!impl_->range_valid)
				{
					impl_->smooth_lo = lo;
					impl_->smooth_hi = hi;
					impl_->range_valid = true;
				}
				else
				{
					impl_->smooth_lo += Impl::kRangeAlpha * (lo - impl_->smooth_lo);
					impl_->smooth_hi += Impl::kRangeAlpha * (hi - impl_->smooth_hi);
				}
				const float span = impl_->smooth_hi - impl_->smooth_lo;
				const float scale = span > 1e-6f ? 1.0f / span : 0.0f;

				// Normaliza para [0,1] e desvira a linha de volta -- simetrico
				// a conversao de entrada, ver o comentario la em cima.
				for(int y = 0; y < n; y++)
				{
					const float *src = raw + (size_t)(n - 1 - y) * n;
					float *dst = impl_->result.data() + (size_t)y * n;
					for(int x = 0; x < n; x++)
					{
						float v = (src[x] - impl_->smooth_lo) * scale;
						dst[x] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
					}
				}
				impl_->has_new_result = true;
			}
		}
	});

	ready_ = true;
	LOGI("NeuralDepth: pronto, entrada %dx%d, delegate %s", kInputW, kInputH,
			impl_->gpu_delegate ? "GPU" : "CPU");
	return true;
#endif
}

void NeuralDepth::SubmitFrame(const uint8_t *rgba)
{
	if(!ready_ || !impl_)
		return;
	{
		std::lock_guard<std::mutex> lock(impl_->frame_mutex);
		impl_->pending_frame.assign(rgba, rgba + (size_t)kInputW * kInputH * 4);
		impl_->has_pending_frame = true;
	}
	impl_->frame_cv.notify_one();
}

bool NeuralDepth::PollResult(float *out_depth)
{
	if(!ready_ || !impl_)
		return false;
	std::lock_guard<std::mutex> lock(impl_->result_mutex);
	if(!impl_->has_new_result)
		return false;
	std::memcpy(out_depth, impl_->result.data(), impl_->result.size() * sizeof(float));
	impl_->has_new_result = false;
	return true;
}

void NeuralDepth::Destroy()
{
	if(!impl_)
		return;
	impl_->running = false;
	impl_->frame_cv.notify_all();
	if(impl_->worker.joinable())
		impl_->worker.join();
#if P5M_WITH_NEURAL_DEPTH
	if(impl_->interpreter)
		TfLiteInterpreterDelete(impl_->interpreter);
	if(impl_->gpu_delegate)
		TfLiteGpuDelegateV2Delete(impl_->gpu_delegate);
	if(impl_->options)
		TfLiteInterpreterOptionsDelete(impl_->options);
	if(impl_->model)
		TfLiteModelDelete(impl_->model);
#endif
	delete impl_;
	impl_ = nullptr;
	ready_ = false;
}

} // namespace p5m
