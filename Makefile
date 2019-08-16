# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/markus/Development/Neuropia

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/markus/Development/Neuropia

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/markus/Development/Neuropia/CMakeFiles /home/markus/Development/Neuropia/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/markus/Development/Neuropia/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named neuropia

# Build rule for target.
neuropia: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 neuropia
.PHONY : neuropia

# fast build rule for target.
neuropia/fast:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/build
.PHONY : neuropia/fast

src/argparse.o: src/argparse.cpp.o

.PHONY : src/argparse.o

# target to build an object file
src/argparse.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/argparse.cpp.o
.PHONY : src/argparse.cpp.o

src/argparse.i: src/argparse.cpp.i

.PHONY : src/argparse.i

# target to preprocess a source file
src/argparse.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/argparse.cpp.i
.PHONY : src/argparse.cpp.i

src/argparse.s: src/argparse.cpp.s

.PHONY : src/argparse.s

# target to generate assembly for a file
src/argparse.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/argparse.cpp.s
.PHONY : src/argparse.cpp.s

src/evotrain.o: src/evotrain.cpp.o

.PHONY : src/evotrain.o

# target to build an object file
src/evotrain.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/evotrain.cpp.o
.PHONY : src/evotrain.cpp.o

src/evotrain.i: src/evotrain.cpp.i

.PHONY : src/evotrain.i

# target to preprocess a source file
src/evotrain.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/evotrain.cpp.i
.PHONY : src/evotrain.cpp.i

src/evotrain.s: src/evotrain.cpp.s

.PHONY : src/evotrain.s

# target to generate assembly for a file
src/evotrain.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/evotrain.cpp.s
.PHONY : src/evotrain.cpp.s

src/idxreader.o: src/idxreader.cpp.o

.PHONY : src/idxreader.o

# target to build an object file
src/idxreader.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/idxreader.cpp.o
.PHONY : src/idxreader.cpp.o

src/idxreader.i: src/idxreader.cpp.i

.PHONY : src/idxreader.i

# target to preprocess a source file
src/idxreader.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/idxreader.cpp.i
.PHONY : src/idxreader.cpp.i

src/idxreader.s: src/idxreader.cpp.s

.PHONY : src/idxreader.s

# target to generate assembly for a file
src/idxreader.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/idxreader.cpp.s
.PHONY : src/idxreader.cpp.s

src/main.o: src/main.cpp.o

.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i

.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s

.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

src/neuropia.o: src/neuropia.cpp.o

.PHONY : src/neuropia.o

# target to build an object file
src/neuropia.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/neuropia.cpp.o
.PHONY : src/neuropia.cpp.o

src/neuropia.i: src/neuropia.cpp.i

.PHONY : src/neuropia.i

# target to preprocess a source file
src/neuropia.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/neuropia.cpp.i
.PHONY : src/neuropia.cpp.i

src/neuropia.s: src/neuropia.cpp.s

.PHONY : src/neuropia.s

# target to generate assembly for a file
src/neuropia.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/neuropia.cpp.s
.PHONY : src/neuropia.cpp.s

src/paralleltrain.o: src/paralleltrain.cpp.o

.PHONY : src/paralleltrain.o

# target to build an object file
src/paralleltrain.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/paralleltrain.cpp.o
.PHONY : src/paralleltrain.cpp.o

src/paralleltrain.i: src/paralleltrain.cpp.i

.PHONY : src/paralleltrain.i

# target to preprocess a source file
src/paralleltrain.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/paralleltrain.cpp.i
.PHONY : src/paralleltrain.cpp.i

src/paralleltrain.s: src/paralleltrain.cpp.s

.PHONY : src/paralleltrain.s

# target to generate assembly for a file
src/paralleltrain.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/paralleltrain.cpp.s
.PHONY : src/paralleltrain.cpp.s

src/params.o: src/params.cpp.o

.PHONY : src/params.o

# target to build an object file
src/params.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/params.cpp.o
.PHONY : src/params.cpp.o

src/params.i: src/params.cpp.i

.PHONY : src/params.i

# target to preprocess a source file
src/params.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/params.cpp.i
.PHONY : src/params.cpp.i

src/params.s: src/params.cpp.s

.PHONY : src/params.s

# target to generate assembly for a file
src/params.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/params.cpp.s
.PHONY : src/params.cpp.s

src/testports.o: src/testports.cpp.o

.PHONY : src/testports.o

# target to build an object file
src/testports.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/testports.cpp.o
.PHONY : src/testports.cpp.o

src/testports.i: src/testports.cpp.i

.PHONY : src/testports.i

# target to preprocess a source file
src/testports.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/testports.cpp.i
.PHONY : src/testports.cpp.i

src/testports.s: src/testports.cpp.s

.PHONY : src/testports.s

# target to generate assembly for a file
src/testports.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/testports.cpp.s
.PHONY : src/testports.cpp.s

src/trainer.o: src/trainer.cpp.o

.PHONY : src/trainer.o

# target to build an object file
src/trainer.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/trainer.cpp.o
.PHONY : src/trainer.cpp.o

src/trainer.i: src/trainer.cpp.i

.PHONY : src/trainer.i

# target to preprocess a source file
src/trainer.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/trainer.cpp.i
.PHONY : src/trainer.cpp.i

src/trainer.s: src/trainer.cpp.s

.PHONY : src/trainer.s

# target to generate assembly for a file
src/trainer.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/trainer.cpp.s
.PHONY : src/trainer.cpp.s

src/trainerbase.o: src/trainerbase.cpp.o

.PHONY : src/trainerbase.o

# target to build an object file
src/trainerbase.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/trainerbase.cpp.o
.PHONY : src/trainerbase.cpp.o

src/trainerbase.i: src/trainerbase.cpp.i

.PHONY : src/trainerbase.i

# target to preprocess a source file
src/trainerbase.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/trainerbase.cpp.i
.PHONY : src/trainerbase.cpp.i

src/trainerbase.s: src/trainerbase.cpp.s

.PHONY : src/trainerbase.s

# target to generate assembly for a file
src/trainerbase.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/trainerbase.cpp.s
.PHONY : src/trainerbase.cpp.s

src/utils.o: src/utils.cpp.o

.PHONY : src/utils.o

# target to build an object file
src/utils.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/utils.cpp.o
.PHONY : src/utils.cpp.o

src/utils.i: src/utils.cpp.i

.PHONY : src/utils.i

# target to preprocess a source file
src/utils.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/utils.cpp.i
.PHONY : src/utils.cpp.i

src/utils.s: src/utils.cpp.s

.PHONY : src/utils.s

# target to generate assembly for a file
src/utils.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/utils.cpp.s
.PHONY : src/utils.cpp.s

src/verify.o: src/verify.cpp.o

.PHONY : src/verify.o

# target to build an object file
src/verify.cpp.o:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/verify.cpp.o
.PHONY : src/verify.cpp.o

src/verify.i: src/verify.cpp.i

.PHONY : src/verify.i

# target to preprocess a source file
src/verify.cpp.i:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/verify.cpp.i
.PHONY : src/verify.cpp.i

src/verify.s: src/verify.cpp.s

.PHONY : src/verify.s

# target to generate assembly for a file
src/verify.cpp.s:
	$(MAKE) -f CMakeFiles/neuropia.dir/build.make CMakeFiles/neuropia.dir/src/verify.cpp.s
.PHONY : src/verify.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... neuropia"
	@echo "... edit_cache"
	@echo "... src/argparse.o"
	@echo "... src/argparse.i"
	@echo "... src/argparse.s"
	@echo "... src/evotrain.o"
	@echo "... src/evotrain.i"
	@echo "... src/evotrain.s"
	@echo "... src/idxreader.o"
	@echo "... src/idxreader.i"
	@echo "... src/idxreader.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
	@echo "... src/neuropia.o"
	@echo "... src/neuropia.i"
	@echo "... src/neuropia.s"
	@echo "... src/paralleltrain.o"
	@echo "... src/paralleltrain.i"
	@echo "... src/paralleltrain.s"
	@echo "... src/params.o"
	@echo "... src/params.i"
	@echo "... src/params.s"
	@echo "... src/testports.o"
	@echo "... src/testports.i"
	@echo "... src/testports.s"
	@echo "... src/trainer.o"
	@echo "... src/trainer.i"
	@echo "... src/trainer.s"
	@echo "... src/trainerbase.o"
	@echo "... src/trainerbase.i"
	@echo "... src/trainerbase.s"
	@echo "... src/utils.o"
	@echo "... src/utils.i"
	@echo "... src/utils.s"
	@echo "... src/verify.o"
	@echo "... src/verify.i"
	@echo "... src/verify.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

