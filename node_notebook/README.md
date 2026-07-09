## 1. Prerequisites

Make sure you have these installed:

### a) Install Deno

```bash
curl -fsSL https://deno.land/install.sh | sh
```

Verify:

```bash
deno --version
```

### b) Install Node.js (LTS)

Even though you’ll use Deno, the Node SDK usually expects Node compatibility.

```bash
node -v
npm -v
```

### c) Install Python + Jupyter

If you don’t already have Jupyter:

```bash
pip install notebook
```

Verify:

```bash
jupyter notebook
```

## 2. Install Deno Kernel for Jupyter

Deno provides an official Jupyter kernel.

### a) Install the Deno Jupyter kernel

```bash
deno install -A -f https://deno.land/x/deno_jupyter/install.ts
```

### b) Register the kernel

```bash
deno jupyter --install
```

You should see something like:

```bash
deno
```

## 3. Start Jupyter Notebook

```bash
jupyter notebook
```

In the browser:

- Click New
- Select Deno as the kernel

## 4.Install & Use the vlmrun Node SDK

### Example: Import Node SDK in Deno Notebook

```ts
import { VlmRun } from "npm:vlmrun";
```

Set the vlmrun API key in `.env` file

```ts
VLMRUN_API_KEY = "your_key_here";
```

Initialize the VLM Run client using the SDK

```ts
const VLMRUN_API_KEY = Deno.env.get("VLMRUN_API_KEY");

const client = new VlmRun({
  apiKey: VLMRUN_API_KEY,
  baseURL: "https://agent.vlm.run/v1",
});
```

### Run a Simple vlmrun Example (Sample)

```ts
const imageUrl = "url of the image...";
const response = await client.agent.completions.create({
  model: "vlm-agent-1",
  messages: [
    {
      role: "user",
      content: [
        { type: "text", text: "Generate a detailed caption for this image" },
        { type: "image_url", image_url: { url: imageUrl } },
      ],
    },
  ],
});

console.log(response.choices[0].message.content);
```
