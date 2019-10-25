# Configuration for packagers
INSTALL_ROOT?=
INSTALL_PREFIX?=/usr/local

# Packaging scripts should not set this variable.
# Maintainers should move towards the configuration
# variables above.
# Settable for compatibility with old package scripts.
INSTALL_PATH?=${INSTALL_ROOT}${INSTALL_PREFIX}

# Local settings
SYSTEMD_SERVICE_PATH?=/usr/lib/systemd
SYSTEMD_SERVICE_NAME=gpu-trace.service
SYSTEMD_SERVICE_IN=${SYSTEMD_SERVICE_NAME}.in
SYSTEMD_SERVICE_PROCESSED:=$(shell mktemp ${SYSTEMD_SERVICE_NAME}.XXXXXX)

all:
	@echo "Run script locally bin/gpu-trace or 'make install'"

install:
	install -m 755 bin/gpu-trace $(INSTALL_PATH)/bin/gpu-trace
	@sed "s|##INSTALL_PREFIX##|${INSTALL_PREFIX}|" ${SYSTEMD_SERVICE_IN} > ${SYSTEMD_SERVICE_PROCESSED}
	install -D -m644 ${SYSTEMD_SERVICE_PROCESSED} ${SYSTEMD_SERVICE_PATH}/system/${SYSTEMD_SERVICE_NAME}
	@rm -f ${SYSTEMD_SERVICE_PROCESSED}
