QT       += core gui

QT +=   opengl xml
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TEMPLATE = app

SOURCES += \
    group_module.cpp \
    wheelDet_trt.cpp

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
OTHER_FILES += ./ballquerypointlayer.cu \
               ./furthestpointsamplelayer.cu \
                ./featureinterpolatedlayer.cu \
                ./grouppointslayer.cu \


CUDA_SOURCES += ./ballquerypointlayer.cu \
                ./furthestpointsamplelayer.cu \
                ./featureinterpolatedlayer.cu \
                ./grouppointslayer.cu \

CUDA_SDK = "/usr/local/cuda/"   # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda/"            # Path to cuda toolkit install

# DO NOT EDIT BEYOND THIS UNLESS YOU KNOW WHAT YOU ARE DOING....

SYSTEM_NAME = ubuntu         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_50           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math


# include paths
INCLUDEPATH += $$CUDA_DIR/include

INCLUDEPATH +=  /home/xx/pytorch/torch/include
INCLUDEPATH += /home/xx/pytorch/torch/include/torch/csrc/api/include  \
INCLUDEPATH += /opt/TensorRT-7.0.0.11/include \
               /opt/TensorRT-7.0.0.11/samples/common \

LIBS += /opt/TensorRT-7.0.0.11/lib/lib*.so \

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib64/

CUDA_OBJECTS_DIR = ./


# Add the necessary libraries
CUDA_LIBS = -lcuda -lcudart

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS
LIBS += /home/xx/pytorch/torch/lib/lib*.so \
        /home/xx/pytorch/torch/lib/lib*.a
# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

HEADERS += \
    cuda_utils.h \
    featureinterpolatedlayer.h \
    grouppointslayer.h \
    logging.h \
    ballquerypointlayer.h \
    furthestpointsamplelayer.h \ \
    wheelDet_trt.h


DISTFILES += \
    ballquerypointlayer.cu \
    featureinterpolatedlayer.cu \
    furthestpointsamplelayer.cu \
    grouppointslayer.cu  \

