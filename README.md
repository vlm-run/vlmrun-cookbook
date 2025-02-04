<div align="center">
<p align="center" style="width: 100%;">
    <img src="https://raw.githubusercontent.com/vlm-run/.github/refs/heads/main/profile/assets/vlm-black.svg" alt="VLM Run Logo" width="80" style="margin-bottom: -5px; color: #2e3138; vertical-align: middle; padding-right: 5px;"><br>
</p>
<h2>VLM Run Cookbook</h2>
<p align="center">
<a href="https://vlm.run"><b>Website</b></a> | <a href="https://docs.vlm.run/"><b>Docs</b></a> | <a href="https://docs.vlm.run/blog"><b>Blog</b></a> | <a href="https://discord.gg/CCY8cYNC"><b>Discord</b></a>
</p>
<p align="center">
<a href="https://discord.gg/4jgyECY4rq"><img alt="Discord" src="https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord"></a>
<a href="https://twitter.com/vlmrun"><img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/vlmrun.svg?style=social&logo=twitter"></a>
</p>
<br>
</div>

Welcome to **VLM Run Cookbook**, a comprehensive collection of examples and notebooks demonstrating the power of structured visual understanding using the VLM Run platform. This repository hosts practical examples and tutorials for extracting structured data from images, videos, and documents using Vision Language Models (VLMs).


### 💡 Why Use This Cookbook?

- 📚 **Practical Examples**: A comprehensive collection of Colab notebooks demonstrating real-world applications of VLM Run.
- 🔋 **Ready-to-Use**: Each example comes with complete code and documentation, making it easy to adapt for your use case.
- 🎯 **Domain-Specific**: Examples cover various domains from financial documents to TV news analysis.
- 💸 **Cost-Effective**: Learn how to optimize your VLM usage for high-volume production scenarios.
- 🛡️ **Enterprise-Ready**: Examples showcase integration with databases and enterprise systems.

### 🚀 Getting Started

To get started with VLM Run and try out these examples:

1. Sign up for early access [here](https://airtable.com/appjX6543bChjNaEN/pagnciKtynSt4rOT9/form)
2. Join our [Discord community](https://discord.gg/CCY8cYNC) for support
3. Open any of our Colab notebooks below to start experimenting


### 📖 Example Notebooks

Our collection of Colab notebooks demonstrates various use cases and integrations:

#### Document Analysis
| **Name** | **Colab** | **Domain** | **Author** | **Last Updated** |
|:---|:---:|:---:|:---:|:---:|
| Financial Presentations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q6CCXzKlEK8nFkvl_z9m0Y1LUL4KwYJ4#scrollTo=YPk5J7YXPxMR) | `document.presentation` | [@spillai](https://github.com/spillai) | 08-13-2024 |
| Hybrid Document Search with ChromaDB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lvHVFDWb6HmeXSypqZm8yY87wdk0_u8f) | `document.presentation` | [@outtanames](https://github.com/outtanames) | 04-04-2024 |
| Hybrid Document Search with MongoDB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xsx3RxX1tmOQFJBQoX6ilRtj5prUhgtS) | `document.presentation` | [@spillai](https://github.com/spillai) | 05-13-2024 |

#### Media Analysis
| **Name** | **Colab** | **Domain** | **Author** | **Last Updated** |
|:---|:---:|:---:|:---:|:---:|
| Financial TV News | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11ELHShm9Buw_StjF9zxAGG9VToDm0gI2) | `tv.news` | [@spillai](https://github.com/spillai) | 08-14-2024 |


### 💡 Understanding Structured Visual Outputs

Vision Language Models (VLMs) like GPT4-Vision and Claude Vision excel at question-answering over visual inputs. However, for automation and integration, we need structured, strongly-typed outputs. VLM Run provides this through predefined schemas for different visual domains. Let's look at an example:

#### Example: Financial Presentation Analysis

<img src="./assets/example-slide.jpeg" alt="drawing" width="600">

#### Structured JSON Output
```json
{
  "description": "The slide presents a 'Differentiated Operating Model' for Selective Insurance, detailing their unique field model, franchise value, and distribution network. It also includes a pie chart showing the 2023 Net Premiums Written, with a total of $4 Billion distributed across different lines of insurance.",
  "title": "Differentiated Operating Model",
  "page_number": 7,
  "plots": [
    {
      "type": "pie",
      "title": "2023 Net Premiums Written",
      "description": "A pie chart showing the distribution of net premiums written by Selective Insurance in 2023, totaling $4 Billion. It is divided into three categories: Standard Commercial Lines (79%), Standard Personal Lines (10%), and Excess and Surplus Lines (11%).",
      "data": "| Category | Percentage | \n| --- | --- | \n| Standard Commercial Lines | 79% | \n| Standard Personal Lines | 10% | \n| Excess and Surplus Lines | 11% |",
      "caption": null
    }
  ],
  "tables": null,
  "others": [
    {
      "data": "### Unique, locally based field model\n- Locally based underwriting, claims, and safety management specialists\n- Proven ability to develop and integrate actionable tools\n- Enables effective portfolio management in an uncertain loss trend environment\n\n### Franchise value distribution model with high-quality partners\n- Approximately 1,550 distribution partners selling our standard lines products and services through approximately 2,650 office locations\n  - ~850 of these distribution partners sell our personal lines products\n  - ~90 wholesale agents sell our E&S business\n  - ~6,400 distribution partners sell National Flood Insurance Program products across 50 states\n\n> \"Everyone with Selective makes our customers feel like the #1 priority. The ease of working with Selective is unmatched.\" - Selective Agent",
      "caption": null,
      "title": null
    }
  ]
}
```

Let's unpack some of the keys in the JSON output above:

<details>
<summary><kbd>"others.data" (formatted)</kbd></summary>

### Unique, locally based field model
- Locally based underwriting, claims, and safety management specialists
- Proven ability to develop and integrate actionable tools
- Enables effective portfolio management in an uncertain loss trend environment

### Franchise value distribution model with high-quality partners
- Approximately 1,550 distribution partners selling our standard lines products and services through approximately 2,650 office locations
  - ~850 of these distribution partners sell our personal lines products
  - ~90 wholesale agents sell our E&S business
  - ~6,400 distribution partners sell National Flood Insurance Program products across 50 states

 "Everyone with Selective makes our customers feel like the #1 priority. The ease of working with Selective is unmatched." - Selective Agent
</details>


<details>
<summary><kbd>"plots.data" (formatted)</kbd></summary>

| Category | Percentage |
| --- | --- |
| Standard Commercial Lines | 79% |
| Standard Personal Lines | 10% |
| Excess and Surplus Lines | 11% |

</details>
<br>

As you can see from the JSON output above, the structured output is designed to be human readable and parseable (with [Pydantic](https://github.com/pydantic/pydantic) / strongly-typed data-structures) for downstream workflow automation. The output is also designed to be easily integrated with traditional databases like Postgres, MongoDB, or even tabular databases like Airtable, where the columns can be automatically mapped to elements in the schema.


### 🔌 Database Integration Examples

The structured outputs from VLM Run are designed to work seamlessly with traditional databases. Our examples demonstrate:

- Storing and indexing structured visual data alongside raw data
- Going beyond simple embeddings with hybrid search capabilities
- Combining structured and unstructured data for advanced retrieval
- Real-world integration patterns with popular databases

Check out our database integration examples in the notebooks above, and join our [Discord](https://discord.gg/CCY8cYNC) to discuss your specific use case!

### 🔗 Quick Links

* 💬 Send us an email at [support@vlm.run](mailto:support@vlm.run) or join our [Discord](https://discord.gg/CCY8cYNC) for help
* 📣 Follow us on [Twitter](https://twitter.com/vlmrun) and [LinkedIn](https://www.linkedin.com/company/vlm-run) to keep up-to-date on our products
* 📚 Check out our [Documentation](https://docs.vlm.run/) for detailed guides and API reference
