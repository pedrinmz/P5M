# Como compilar — o caminho mais simples

## Primeiro, um problema que o zip que você me mandou tem

O `P5M-0.2.0-beta.zip` original é um download de "baixar ZIP" do GitHub, e
esse tipo de download **não traz o conteúdo de submódulos git** — só a pasta
vazia onde o submódulo deveria estar. Este projeto depende do `chiaki-ng`
(quem fala com o PS5 de verdade) como submódulo em `external/chiaki-ng/`, e
sem ele o `CMakeLists.txt` recusa até configurar:

```
Submodulo chiaki-ng ausente em .../external/chiaki-ng. Rode: git submodule update --init --recursive
```

A correção é simples: **clonar com git em vez de baixar o zip**, uma vez só.
Depois disso as edições deste pacote entram por cima, copiando os arquivos.

## Passo 1 — Instalar o Android Studio

Baixe em https://developer.android.com/studio e instale normal (próximo,
próximo, próximo). Ele já vem com o JDK — não precisa instalar Java
separado.

Na primeira abertura, ele pode pedir pra baixar o SDK do Android — deixe.

## Passo 2 — Clonar o projeto de verdade

Abra um terminal (dentro do próprio Android Studio tem um: "View → Tool
Windows → Terminal") e rode:

```bash
git clone --recursive https://github.com/beecrepaldi-afk/P5M
cd P5M
```

O `--recursive` é o que baixa o `chiaki-ng` junto. Se esquecer, dá pra
corrigir depois com `git submodule update --init --recursive` dentro da
pasta.

## Passo 3 — Copiar as mudanças deste pacote por cima

Do zip que eu te devolvi (`P5M-0.2.0-beta-3d-neural.zip`), copie estes
arquivos para dentro da pasta que você acabou de clonar, **substituindo os
originais**:

```
app/build.gradle
app/src/main/cpp/CMakeLists.txt
app/src/main/cpp/tone_mapper.h
app/src/main/cpp/tone_mapper.cpp
app/src/main/cpp/neural_depth.h        (novo)
app/src/main/cpp/neural_depth.cpp      (novo)
app/src/main/cpp/xr_session.h
app/src/main/cpp/xr_session.cpp
app/src/main/assets/midas_v21_small_256_fp16.tflite   (novo)
docs/3D-NEURAL-INTEGRACAO.md           (novo)
docs/COMO-COMPILAR-3D-NEURAL.md        (novo, este arquivo)
docs/convert_midas_referencia.py       (novo)
```

No Windows/Mac, o jeito mais simples é arrastar os arquivos do zip
extraído pra cima da pasta clonada no Explorador de Arquivos / Finder, e
mandar substituir quando perguntar. No terminal, dá pra fazer com `cp -r`
(Mac/Linux) ou `xcopy /Y` (Windows) — mas arrastar é mais difícil de errar.

## Passo 4 — Abrir no Android Studio

`File → Open`, aponte para a pasta `P5M` que você clonou (a que tem o
`settings.gradle` dentro, na raiz).

Na primeira vez, o Android Studio vai demorar um pouco — ele vai baixar
sozinho: o SDK na versão que o projeto pede (34), o NDK na versão exata que
o `CMakeLists.txt` espera (`26.3.11579264`), o CMake, e as bibliotecas do
`build.gradle` — TFLite incluído, já que deixei a dependência adicionada.
Isso tudo acontece numa barra de progresso embaixo, chamada "Gradle Sync".
Espere ela terminar antes de mexer em mais nada.

Se ela terminar com erro pedindo pra instalar alguma versão de NDK/SDK
específica, geralmente tem um link azul clicável tipo "Install missing NDK
and sync project" — clique nele, é automático.

## Passo 5 — Preparar o Quest 3

No celular, no app **Meta Quest**: Aparelho → seu Quest 3 → Recursos para
desenvolvedores → ative o Modo de Desenvolvedor (isso exige ter uma conta de
desenvolvedor Meta, gratuita — se ainda não tem, o próprio app te guia na
hora).

Depois, conecte o Quest 3 no computador com um cabo USB-C (o de carregar
mesmo serve, contanto que transfira dados). Coloque o headset, vai aparecer
um aviso pedindo permissão de depuração USB — aceite, e marque "sempre
permitir deste computador" pra não perguntar de novo.

## Passo 6 — Rodar

De volta no Android Studio: no topo, tem um menu suspenso de dispositivo
(ao lado do botão verde de play ▶). Clique nele — o Quest 3 deveria aparecer
na lista (como "Quest 3" ou o nome que você deu a ele). Selecione.

Clique no ▶ verde. Isso compila, instala e abre o app dentro do headset
sozinho — sem precisar mexer em `adb` nem gerenciar arquivo `.apk` na mão.
Na primeira vez, a compilação do código nativo (C++) demora alguns minutos;
depois disso, só o que mudou recompila, então fica rápido.

## Se preferir terminal em vez do botão

Funciona igual, é como o próprio README do projeto já documenta:

```bash
./gradlew assembleDebug
```

Isso gera um `.apk` em `app/build/outputs/apk/debug/`. Pra instalar no
Quest 3 conectado (com os mesmos passos 5 acima já feitos):

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

(`adb` vem junto do Android Studio, mas se quiser usar sem abrir o Studio,
precisa adicionar ele ao PATH do sistema — o instalador geralmente oferece
essa opção.)

## Se der erro

- **"submodule ausente"** de novo, mesmo depois do clone `--recursive`: rode
  `git submodule update --init --recursive` dentro da pasta do projeto.
- **Erro de `find_package(tensorflowlite_c...)` não encontrado**: o Gradle
  ainda não baixou o AAR do TFLite. Sincronize o projeto de novo (ícone do
  elefante com uma seta, ou `File → Sync Project with Gradle Files`) e tente
  compilar de novo.
- **Qualquer outro erro de compilação do C++**: copie a mensagem de erro
  completa (geralmente aparece na aba "Build" embaixo) — com isso dá pra
  saber exatamente o que corrigir, em vez de adivinhar.
