cmake_minimum_required(VERSION 3.22)
project(p5m)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# libchiaki + chiaki-jni vem do submodulo, com as mesmas flags que o projeto
# Android do chiaki-ng usa. Nao tocamos em nada la dentro.
set(CHIAKI_NG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../external/chiaki-ng")
get_filename_component(CHIAKI_NG_DIR "${CHIAKI_NG_DIR}" ABSOLUTE)
if(NOT EXISTS "${CHIAKI_NG_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Submodulo chiaki-ng ausente em ${CHIAKI_NG_DIR}. Rode: git submodule update --init --recursive")
endif()
add_subdirectory("${CHIAKI_NG_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/chiaki-ng")

# Loader OpenXR distribuido como AAR do Khronos, exposto via prefab.
find_package(OpenXR REQUIRED CONFIG)

add_library(p5m-vr SHARED
        pilha_nativa.cpp
        p5m_jni.cpp
        spatializer.cpp
        tone_mapper.cpp
        neural_depth.cpp
        xr_session.cpp)

# -- 3D neural (pausado) ----------------------------------------------------
#
# neural_depth.cpp compila e funciona sem isto -- P5M_WITH_NEURAL_DEPTH so
# liga quando <tensorflow/lite/c/c_api.h> existe (ver o __has_include no topo
# de neural_depth.h), e sem ele a classe inteira vira no-ops e o 3D fica na
# estimativa heuristica, que e onde estamos agora de proposito.
#
# find_package(tensorflowlite_c ...) ficava aqui, REQUIRED. Voltou a ficar de
# fora: o AAR oficial do TFLite (org.tensorflow:tensorflow-lite) nao publica
# um pacote CMake utilizavel por find_package via prefab -- e uma lacuna
# conhecida e sem solucao oficial (ver as issues abertas no repositorio do
# tensorflow sobre "prefab" + "tensorflowlite_c", 2023 em diante). Exigir
# aqui derrubava o build inteiro por causa de uma peca que nunca ia se
# encaixar assim. Quando isso for retomado, o caminho provavel nao e mais
# find_package: e rodar a inferencia do lado Kotlin (via
# org.tensorflow.lite.Interpreter, o caminho realmente suportado) e so trocar
# dados com o C++ por JNI -- reaproveitando a fila de quadros e a textura de
# resultado que ja existem aqui, so a chamada ao interpretador muda de lado.
target_include_directories(p5m-vr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

find_library(ANDROID_LIB_LOG log)
find_library(ANDROID_LIB_ANDROID android)
find_library(ANDROID_LIB_EGL EGL)
find_library(ANDROID_LIB_GLESV3 GLESv3)
# Painel de ajuda: o texto e desenhado com Canvas do lado Java e chega aqui
# como android.graphics.Bitmap.
find_library(ANDROID_LIB_JNIGRAPHICS jnigraphics)

target_link_libraries(p5m-vr
        OpenXR::openxr_loader
        tensorflowlite_c::tensorflowlite_c
        tensorflowlite_gpu::tensorflowlite_gpu_delegate
        "${ANDROID_LIB_LOG}"
        "${ANDROID_LIB_ANDROID}"
        "${ANDROID_LIB_EGL}"
        "${ANDROID_LIB_GLESV3}"
        "${ANDROID_LIB_JNIGRAPHICS}")
