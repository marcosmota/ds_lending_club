.PHONY: install
install:
	@echo "Install..."
	@uv sync
	@echo "Done!"

.PHONY: donwload-dataset
donwload-dataset:
	@echo "Downloading the dataset..."
	@gdown --folder https://drive.google.com/drive/folders/1NVQtamDRuGuAIZCRfidvnlSPH1-xLGlG
	@echo "Done!"
