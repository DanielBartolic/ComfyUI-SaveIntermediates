# ComfyUI-SaveIntermediates

Save and stream intermediate sampling steps during diffusion. Perfect for showing generation progress on your frontend instead of a loading spinner.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DanielBartolic/ComfyUI-SaveIntermediates.git
```

## Usage

### For Serverless/API Streaming

Connect the node between your model loader and sampler:

```
Load Checkpoint → [Save Intermediate Steps] → ClownSampler → ...
                         ↓
                   (outputs to progress folder)
```

Your frontend can poll `progress.json` to get:
- Current step / total steps
- Progress percentage
- Path to latest image
- Optional base64 encoded image

### Workflow with Image Output

```
Load Checkpoint → [Save Intermediate Steps] → ClownSampler → [Get Intermediate Images] → Preview/Save
                                                    ↓
                                              (latent output)
```

## Nodes

### Save Intermediate Steps
Wraps model and saves TAESD previews at each step.

**Inputs:**
- `model` - From your model loader
- `steps` - Total sampling steps (for progress calculation)
- `job_id` - Unique ID for this job (creates subfolder)
- `output_folder` - Base folder name (default: "progress")
- `filename_prefix` - Prefix for files (default: "step")
- `format` - jpeg or png
- `include_base64` - Include base64 in progress.json

**Output:**
- `model` - Pass to your sampler

### Save Intermediate Steps (Advanced)
Same as above with filtering options:
- `save_every_n` - Save every Nth step
- `start_step` / `end_step` - Range of steps to save

### Get Intermediate Images
Collects all saved intermediates as IMAGE batch after sampling.

**Inputs:**
- `latent` - Connect from sampler (ensures this runs after)
- `job_id` - Match the job_id used in Save node
- `output_folder` - Match the folder used

**Output:**
- `images` - Batch of all intermediate images

## Output Structure

```
ComfyUI/output/progress/{job_id}/
├── progress.json      # Poll this for status
├── latest.jpeg        # Always the most recent step
├── step_0000.jpeg
├── step_0001.jpeg
├── step_0002.jpeg
...
```

### progress.json Format

```json
{
  "job_id": "abc123",
  "status": "generating",
  "current_step": 15,
  "total_steps": 20,
  "progress_percent": 75,
  "latest_image": "step_0015.jpeg",
  "timestamp": 1704567890.123,
  "image_base64": "..."  // if include_base64 enabled
}
```

## Frontend Integration Example

```javascript
// Poll for progress
async function pollProgress(jobId) {
  const response = await fetch(`/output/progress/${jobId}/progress.json`);
  const data = await response.json();

  if (data.status === 'generating') {
    // Show preview image
    const imgUrl = `/output/progress/${jobId}/${data.latest_image}`;
    updatePreview(imgUrl);
    updateProgressBar(data.progress_percent);
  }

  if (data.status !== 'completed') {
    setTimeout(() => pollProgress(jobId), 500);
  }
}
```

## License

MIT
