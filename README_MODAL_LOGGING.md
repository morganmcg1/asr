# Modal Logging Instructions

## Streaming Container Logs

Modal can stream container logs back to your terminal when running training jobs. Here's how:

### 1. Using `modal run` (Simplest)

When you start training with:
```bash
modal run modal_parakeet_final.py --max-epochs 3 --batch-size 8 --subset-size 100
```

Anything your function prints to stdout/stderr (`print`, Python `logging`) is automatically streamed to your terminal. This is the default behavior of `modal run`.

### 2. Using Python Driver Scripts

If you're using a Python script that drives Modal (e.g., `app.run()`), wrap it with `modal.enable_output()`:

```python
import modal

app = modal.App()

@app.function(gpu="H100")
def train():
    print("Training logs...")
    
if __name__ == "__main__":
    with modal.enable_output():  # Enable log streaming
        with app.run():
            train.remote()
```

### 3. Tailing Logs from Running Jobs

For already-running or detached jobs:

**Per app (recommended for ML runs):**
```bash
modal app logs my-app-name
# or with app ID:
modal app logs ap-123456
```

**Per container (fine-grained):**
```bash
modal container list              # get CONTAINER_ID
modal container logs ta-xxxxx     # stream logs for that container
```

## Modal Authentication

To authenticate with Modal:

```bash
modal token set --token-id <YOUR_TOKEN_ID> --token-secret <YOUR_TOKEN_SECRET> --profile=weightsandbiases
modal profile activate weightsandbiases
```

**Security Note:** The token credentials are stored in your repository's secrets and should never be committed to version control.

## Best Practices

1. **Always use `modal run`** for interactive training - logs stream automatically
2. **Use `modal.enable_output()`** if you're orchestrating via Python `app.run()`  
3. **Use `modal app logs`** to check on detached/deployed runs
4. **Monitor W&B dashboard** for training metrics and progress

## References

- [Modal enable_output docs](https://modal.com/docs/reference/modal.enable_output)
- [Modal app logs CLI](https://modal.com/docs/reference/cli/app)
- [Modal container logs CLI](https://modal.com/docs/reference/cli/container)
