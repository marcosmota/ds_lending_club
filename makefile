.PHONY: install
install:
	@echo "Install..."
	@uv sync
	@echo "Done!"

.PHONY: download-dataset
download-dataset:
	@echo "Downloading the dataset..."
	@gdown --folder https://drive.google.com/drive/folders/1NVQtamDRuGuAIZCRfidvnlSPH1-xLGlG
	@echo "Done!"
