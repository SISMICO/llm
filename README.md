# 🏠 Local AI Ecosystem Guide (2026)
This setup allows you to run a single Large Language Model (LLM) on a **Raspberry Pi** and connect it to your IDEs, Terminal, and Web Browser across your local network.

---

## 🧠 1. The Server: llama.cpp
**Description:** The core engine. It runs the model and provides an API that mimics OpenAI and Anthropic natively.
* **Link:** [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
* **Installation (on Raspberry Pi):**
    1. **Install dependencies:** 
       ```bash
       sudo apt update && sudo apt install git cmake g++ build-essential libcpp-httplib-dev -y
       ```
    2. **Clone & Build:**
       ```bash
       git clone [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
       cd llama.cpp && mkdir build && cd build
       cmake .. -DGGML_NATIVE=ON
       cmake --build . --config Release -j4
       ```
    3. **Run the Server:**
       ```bash
       ./bin/llama-server -m Qwen_Qwen3.5-35B-A3B-Q3_K_M.gguf --host 0.0.0.0 --port 8080 --jinja --ctx-size 32768 --flash-attn on --reasoning-budget 0
       ```

---

## 💻 2. IntelliJ IDEA: ProxyAI
**Description:** The premier JetBrains plugin for local LLMs. It features "Auto-Apply" for streaming code changes and deep project indexing.
* **Link:** [ProxyAI on JetBrains Marketplace](https://plugins.jetbrains.com/plugin/21056-proxyai)
* **Installation:**
    1. Open IntelliJ → `Settings` → `Plugins`.
    2. Search for **ProxyAI** and click **Install**.
    3. Go to `Settings` → `Tools` → `ProxyAI`.
    4. Select **OpenAI Compatible** (or Custom Provider).
    5. Set **Endpoint** to: `http://<PI_IP_ADDRESS>:8080/v1`
    6. Set **API Key** to: `sk-dummy` (anything works).

---

## ⚡ 3. VS Code: Continue.dev
**Description:** A powerful open-source autopilot. It provides tab-autocomplete, a chat sidebar, and inline "Edit" commands.
* **Link:** [continue.dev](https://www.continue.dev/)
* **Installation:**
    1. Install the **Continue** extension from the VS Code Marketplace.
    2. Click the **gear icon (⚙️)** in the Continue sidebar to open `config.json`.
    3. Add this entry to the `models` array:
       ```json
       {
         "name": "Raspberry Pi LLM",
         "provider": "openai",
         "model": "Qwen_Qwen3.5-35B-A3B-Q3_K_M.gguf",
         "apiBase": "http://<PI_IP_ADDRESS>:8080/v1"
       }
       ```

---

## 🛠️ 4. Terminal Agent: Claude Code
**Description:** Anthropic’s official CLI agent. Because `llama-server` supports the Anthropic API, Claude Code can run against your Pi to fix bugs and run tests.
* **Link:** [claude.ai/download](https://claude.ai/download)
* **Installation:**
    1. **Install CLI:** 
       ```bash
       curl -fsSL https://claude.ai/install.sh | bash
       ```
    3. **Configure Environment:** Add these to your `.zshrc` or `.bashrc`:
       ```bash
       export ANTHROPIC_BASE_URL="http://<PI_IP_ADDRESS>:8080"
       export ANTHROPIC_API_KEY="ollama"
       ```
    4. **Launch:** Run `claude` in any project folder.

---

## 🌐 5. Web Interface: Open WebUI
**Description:** A polished, ChatGPT-style web interface accessible via browser on any device in your network.
* **Link:** [openwebui.com](https://openwebui.com)
* **Installation (via Docker):**
    ```bash
    docker run -d -p 3000:8080 \
      -e OPENAI_API_BASE_URL="http://host.docker.internal:8080/v1" \
      -e OPENAI_API_KEY="sk-dummy" \
      --add-host=host.docker.internal:host-gateway \
      --name open-webui ghcr.io/open-webui/open-webui:main
    ```
    *Access via `http://localhost:3000`*

---

## ⚙️ 6. CLI Command Helper: AIChat
**Description:** Generates shell commands, bash scripts, and git commits from plain English.
* **Link:** [github.com/sigoden/aichat](https://github.com/sigoden/aichat/releases)
* **Installation:**
    1. Download from https://github.com/sigoden/aichat/releases
    2. Extract to you PATH directory
    3. Configure  `~/.config/aichat/config.yaml` file:
       ```
       clients:
         - type: openai
           api_base: http://localhost:8080/v1
           api_key: sk-dummy
           models:
             - name: Qwen_Qwen3.5-35B-A3B-Q3_K_M.gguf
       ```
