#!/bin/bash
# Download MetaboLM pretrained weights from Figshare
#
# NOTE: There are two relevant Figshare links. Verify which URL works before
# running unattended. The private share link may expire or change.
#
# Pre-training weights (private share link):
#   https://figshare.com/s/bd69f74785946802b585
#
# Fine-tuning weights (DOI 10.6084/m9.figshare.30744284):
#   https://figshare.com/ndownloader/articles/30744284/versions/1
#
# The script below attempts the DOI-based URL first. If the download produces
# an HTML page instead of a .pth file, try the private share link or download
# manually from the Figshare web UI.

set -e
cd "$(dirname "$0")/.."
mkdir -p weights

echo "Downloading MetaboLM pre-training weights from Figshare..."

# Primary: DOI-based download (article 30744284, version 1)
DOWNLOAD_URL="https://figshare.com/ndownloader/articles/30744284/versions/1"

# Alternative: private share link (uncomment if the above does not work)
# DOWNLOAD_URL="https://figshare.com/s/bd69f74785946802b585"

wget -O weights/pretrained_ckpt.pth "$DOWNLOAD_URL"

echo ""
echo "Download complete. File saved to weights/pretrained_ckpt.pth"
echo ""
echo "Verify with:"
echo "  python -c \"import torch; ckpt = torch.load('weights/pretrained_ckpt.pth', map_location='cpu'); print(type(ckpt)); print(ckpt.keys() if hasattr(ckpt, 'keys') else 'not a dict')\""
echo ""
echo "If the file is a zip archive (Figshare article download), unzip it first:"
echo "  cd weights && unzip pretrained_ckpt.pth && cd .."
