run: # should auto identify for MSEAS cluster or local
	python3 -m qg.deploy.with_compute
	
multiple:
	python3 -m qg.deploy.multiple

install:
	bash install/install.sh