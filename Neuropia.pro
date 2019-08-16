QT -= gui

TARGET = Neuropia
CONFIG += console
CONFIG -= app_bundle

!msvc:QMAKE_CXXFLAGS += -Wno-multichar

mingw-64:DEFINES += STD_ALLOCATOR #mingw-64 has bug in thread locale: https://github.com/ChaiScript/ChaiScript/issues/402

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

INCLUDEPATH += include

TEMPLATE = app

CONFIG += c++17

SOURCES += src/main.cpp \
    src/neuropia.cpp \
    src/idxreader.cpp \
    src/neuropia_simple.cpp \
    src/utils.cpp \
    src/testports.cpp \
    src/params.cpp \
    src/trainer.cpp \
    src/verify.cpp \
    src/paralleltrain.cpp \
    src/evotrain.cpp \
    src/argparse.cpp \
    src/trainerbase.cpp

HEADERS += \
    include/default.h \
    include/neuropia.h \
    include/idxreader.h \
    include/matrix.h \
    include/neuropia_simple.h \
    include/paralleltrain.h \
    include/utils.h \
    include/evotrain.h \
    include/verify.h \
    include/params.h \
    include/trainer.h \
    include/argparse.h \
    include/trainerbase.h

DISTFILES +=    \
    tests/tests.txt   \
    tests/tests1.txt \
    tests/tests2.txt  \
    tests/tests23.txt \
    tests/tests3.txt  \
    tests/tests5.txt  \
    tests/tests4.txt  \
    tests/tests7.txt \
    tests/tests8.txt \
    tests/tests9.txt \
    tests/tests10.txt \
    tests/tests11.txt \
    tests/tests12.txt \
    tests/tests13.txt \
    tests/tests14.txt \
    tests/tests15.txt \
    tests/tests16.txt \
    tests/tests17.txt \
    tests/tests18.txt \
    tests/tests20.txt \
    tests/tests21.txt \
    tests/tests22.txt

