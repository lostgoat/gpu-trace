INSTALL_PATH?=/usr/local

all:
	@echo "Run script locally bin/amdgpu-trace.sh or 'make install'"

install:
	install -m 755 bin/amdgpu-trace $(INSTALL_PATH)/bin/amdgpu-trace
