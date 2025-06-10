# 🤖 Model Information

## Automatic Model Download

**The Whisper models are NOT included in this repository** due to GitHub's 100MB file size limit.

### What happens when you run the app:

1. **First time**: Models will be automatically downloaded to `./models/` directory
2. **Subsequent runs**: Models are loaded from local cache (much faster!)

### Model Sizes:

| Model    | Size   | Speed     | Accuracy |
| -------- | ------ | --------- | -------- |
| tiny     | 39 MB  | Very Fast | Good     |
| base     | 74 MB  | Fast      | Good     |
| small    | 244 MB | Medium    | Better   |
| medium   | 769 MB | Medium    | Better   |
| large-v3 | ~3GB   | Slower    | Best     |

### Storage Requirements:

- **Quick Start** (medium): ~800MB
- **Full System** (large-v3): ~3GB

### Download Time:

- First run: 2-10 minutes (depending on internet speed)
- The models are cached locally for future use

### Manual Model Management:

```bash
# View downloaded models
ls -la models/

# Clear model cache (if needed)
rm -rf models/

# Models will re-download on next run
```

**Note**: The `models/` directory is in `.gitignore` to keep the repository lightweight while ensuring the app works perfectly!
