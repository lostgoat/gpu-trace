all:
	@echo "Run script locally bin/amdgpu-trace.sh or 'make install'"

install:
	cp bin/amdgpu-trace /usr/local/bin/amdgpu-trace
	chmod 0755 /usr/local/bin/amdgpu-trace
