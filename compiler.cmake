function (SET_COMPILER_FLAGS)

    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()

    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

    set(PEDANTIC_WARNINGS FALSE)
    set(SANITIZER FALSE)
    set(SANITIZER_THREAD FALSE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(PEDANTIC_WARNINGS TRUE)
        if(${CMAKE_BUILD_TYPE} STREQUAL "RELWITHDEBINFO")
            set(SANITIZER TRUE)
        endif()
    endif()  


    if(PEDANTIC_WARNINGS )
        target_compile_options(${PROJECT_NAME} PRIVATE 
            -pedantic
            -Wall 
            -Wextra 
            -Wcast-align 
            -Wcast-qual 
            -Wctor-dtor-privacy 
            -Wdisabled-optimization 
            -Wformat=2 
            -Winit-self 
            -Wlogical-op 
            -Wmissing-declarations 
            -Wmissing-include-dirs 
            -Wnoexcept 
            -Wold-style-cast  
            -Woverloaded-virtual 
            -Wredundant-decls 
            -Wshadow 
            -Wsign-conversion 
            -Wsign-promo 
            -Wstrict-null-sentinel 
            -Wstrict-overflow=2 # 3-5 end up errorneous warning with Data::writeHeader 
            -Wswitch-default 
            -Wundef 
            -Werror 
            -Wno-unused
            # not applicable -Wabi
            # not an issue due c++17 copy elision -Waggregate-return
            -Wconversion 
            -Weffc++
            # There is some header only-code -Winline
            # Maintaining manual padding is pia, assumed not real benefit - -Wpadded
            -Wswitch-enum
            -Wunsafe-loop-optimizations
            -Wzero-as-null-pointer-constant
            #-Wuseless-cast cmake defined types wont be happy with types
        )
    endif()

    if(SANITIZER)
        message(WARNING "Sanitizers enabled")
        set(ASAN_FLAGS
            -fsanitize=address
            -fsanitize=pointer-compare
            -fsanitize=pointer-subtract
            -fsanitize=leak
            -fsanitize=undefined
            -fsanitize=shift
            -fsanitize=shift-exponent
            -fsanitize=shift-base
            -fsanitize=integer-divide-by-zero
            -fsanitize=unreachable
            -fsanitize=vla-bound
            -fsanitize=null
            -fsanitize=return
            -fsanitize=signed-integer-overflow
            -fsanitize=bounds
            -fsanitize=bounds-strict
            -fsanitize=alignment
            -fsanitize=object-size
            -fsanitize=float-divide-by-zero
            -fsanitize=float-cast-overflow
            -fsanitize=nonnull-attribute
            -fsanitize=returns-nonnull-attribute
            -fsanitize=bool
            -fsanitize=enum
            -fsanitize=vptr
            -fsanitize=pointer-overflow
            -fsanitize=builtin)
        add_compile_options(${ASAN_FLAGS})
        add_link_options(${ASAN_FLAGS})
    endif()

    if(SANITIZER_THREAD)
        if(SANITIZER)
            message(FATAL_ERROR "SANITIZER_THREAD cannot co exists with SANITIZER flag!")
        endif()
        message(WARNING "Thread sanitizers enabled")
        set(ASAN_FLAGS -fsanitize=thread) 
        add_compile_options(${ASAN_FLAGS})
        add_link_options(${ASAN_FLAGS})
    endif()


endfunction()