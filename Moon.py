import math
import os
import numpy
import threading
import ast
import numbers
import abc
import json
import time
import random
import sys
import glob
import subprocess
import shutil
import traceback
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Dict, Any, Optional, Union, Literal
import ollama
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.theme import Theme
from rich.status import Status
from rich.table import Table
from rich.align import Align
from rich.syntax import Syntax
from rich import box
from rich.markdown import Markdown
from rich.status import Status
from rich.color import Color
import contextlib
from pypdf import PdfReader

# Strict Monochromatic Theme: White, Grey variants, and Red for errors only.
CODER_THEME = Theme({
    "primary": "white",
    "dim": "grey37",
    "subtle": "grey70",
    "error": "red",
    "accent": "white",
    "border": "grey37",
    "bg": "black",
    "link": "white underline",
    "code": "grey70",          # Inline code
    "strong": "white bold",    # Bold
    "em": "grey70 italic",     # Italics
    "heading.1": "white bold",
    "heading.2": "white bold",
    "heading.3": "white bold",
})

CODER_GRAPHIC = [
    (" ███╗   ███╗ ██████╗  ██████╗ ███╗   ██╗", "#0000ff"),
    (" ████╗ ████║██╔═══██╗██╔═══██╗████╗  ██║", "#3333ff"),
    (" ██╔████╔██║██║   ██║██║   ██║██╔██╗ ██║", "#5555ff"),
    (" ██║╚██╔╝██║██║   ██║██║   ██║██║╚██╗██║", "#7777ff"),
    (" ██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║ ╚████║", "#9999ff"),
    (" ╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝", "#9999ff"),
]

console = Console(theme=CODER_THEME)

class ToolCall(BaseModel):
    """Schema for a tool execution."""
    tool_name: str = Field(description="The name of the tool to use")
    arguments: Dict[str, Any] = Field(description="The arguments for the tool as a dictionary")

class AgentResponse(BaseModel):
    """The mandatory structure for every LLM response."""
    trajectory: str = Field(description="Internal roadmap: [Step X/Y | Task -> Next Target]")
    status_summary: str = Field(description="A punchy, 3-5 word UI status update. E.g., 'Scanning filesystem', 'Refining logic', 'Compiling report'. No pipes, no arrows.")
    synthesis: str = Field(description="Deep reasoning or error analysis")
    action: Optional[ToolCall] = Field(description="The tool to call, or null if finishing")
    completion_summary: Optional[str] = Field(description="Final technical summary")

class AgentState(TypedDict):
    mission: str
    trajectory: str
    synthesis: str
    last_observation: str
    history: List[Dict[str, str]]
    next_step: str
    current_node: str
    error_count: int
    is_finished: bool
    pending_action: Optional[ToolCall]
    consecutive_reasoning_steps: int

# --- 1. MINIMALIST ARCHITECTURAL UI ---

class RichUI:
    """A high-end monochromatic interface using Rich for structural beauty."""
    
    def __init__(self):
        self.console = Console(theme=CODER_THEME, force_terminal=True)
        self.status_msg = "Idle"

    @contextlib.contextmanager
    def node_context(self, name: str, is_error: bool = False):
        # Professional character set
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        # Color cycle for the spinner to simulate a 'CSS transition' effect
        spinner_colors = ["#333333", "#555555", "#777777", "#999999", "#999999"]
        
        tag_style = "red on grey37" if is_error else "white on grey37"
        node_tag = Text(f" {name.upper()} ", style=tag_style)
        separator = Text(" ⸻ ", style="dim")
        
        class ContextState:
            def __init__(self):
                self.message = ""
                self.spinner_idx = 0
                self.running = True

        state = ContextState()

        with Live(Text("", style="subtle"), console=self.console, refresh_per_second=20) as live:
            def animate():
                while state.running:
                    if state.message:
                        spinner = spinners[state.spinner_idx % len(spinners)]
                        # Simulate CSS animation by cycling spinner colors
                        color = spinner_colors[(state.spinner_idx // 2) % len(spinner_colors)]
                        
                        display_line = Text()
                        display_line.append(node_tag)
                        display_line.append(separator)
                        display_line.append(state.message, style="white")
                        display_line.append(f" {spinner}", style=color) # Animated color
                        
                        live.update(display_line)
                        state.spinner_idx += 1
                    else:
                        live.update(node_tag + separator + Text("Initializing...", style="dim"))
                    
                    time.sleep(0.08)

            anim_thread = threading.Thread(target=animate, daemon=True)
            anim_thread.start()

            class StatusProxy:
                def update(self, msg: str):
                    state.message = msg.strip().replace('\n', ' ')

            proxy = StatusProxy()
            try:
                yield proxy
                state.running = False
                anim_thread.join(timeout=0.1)
                
                # --- NODE FINISHED SUCCESSFULLY ---
                if not is_error:
                    success_text = Text()
                    success_text.append(node_tag)
                    success_text.append(separator)
                    success_text.append("✅", style="white")
                    live.update(success_text)
                else:
                    fail_text = Text()
                    fail_text.append(node_tag)
                    fail_text.append(separator)
                    fail_text.append("❌ Failed", style="red")
                    live.update(fail_text)
                    
            except Exception as e:
                state.running = False
                anim_thread.join(timeout=0.1)
                live.update(node_tag + separator + Text("System Error", style="red"))
                raise e


    def _force_clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        self.console.clear()

    def startup_sequence(self):

        self._force_clear()

    def boot(self):
        self._force_clear()
        
        # 1. Render the Heavy Graphic Block
        for line, style in CODER_GRAPHIC:
            self.console.print(f"[{style}]{line}[/{style}]")

    
    def _get_key(self):
        """Cross-platform non-blocking key reader for arrow keys and enter."""
        if os.name == 'nt':  # Windows
            import msvcrt
            ch = msvcrt.getch()
            if ch in (b'\x00', b'\xe0'):  # Special key prefix
                ch = msvcrt.getch()
                if ch == b'H': return 'up'
                if ch == b'P': return 'down'
            elif ch == b'\r': return 'enter'
        else:  # Linux / MacOS
            import tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # Escape sequence
                    next_chars = sys.stdin.read(2)
                    if next_chars == '[A': return 'up'
                    if next_chars == '[B': return 'down'
                elif ch in ('\r', '\n'): return 'enter'
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return None

    
    def select_model(self):
        """Interactive model selection using arrow keys and Enter."""
        self._force_clear()
        
        try:
            # 1. Fetch models from Ollama
            output = subprocess.check_output(['ollama', 'list'], text=True)
            lines = output.strip().split('\n')
            if len(lines) <= 1: 
                self.console.print("[error]NO MODELS DETECTED IN SYSTEM[/error]")
                return None
            models = [line.split()[0] for line in lines[1:] if line.split()]

            selected_idx = 0

            # 2. Define the UI Generator (to be used by Live)
            def generate_menu():
                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("ID", justify="right", style="dim")
                table.add_column("MODEL NAME", justify="left", style="white")

                for i, model in enumerate(models):
                    is_selected = (i == selected_idx)
                    prefix = "❯ " if is_selected else "  "
                    style = "primary bold" if is_selected else "dim"
                    
                    # Use the prefix and style to indicate selection
                    table.add_row(
                        f"{prefix}{i+1:02d}", 
                        f"[{style}]{model}[/{style}]"
                    )
                return table

            # 3. The Interactive Loop using Rich Live
            self.console.print("\n[#0000ff]MODEL SELECTION[/#0000ff]\n")
            
            with Live(generate_menu(), console=self.console, refresh_per_second=15) as live:
                while True:
                    key = self._get_key()
                    
                    if key == 'up':
                        selected_idx = (selected_idx - 1) % len(models)
                    elif key == 'down':
                        selected_idx = (selected_idx + 1) % len(models)
                    elif key == 'enter':
                        break
                    elif key is None and key != '': # Ignore other keys
                        pass
                    
                    # Update the live display
                    live.update(generate_menu())

            chosen_model = models[selected_idx]
            self.console.print(f"\n[dim]PROCEEDING WITH {chosen_model.upper()}[/dim]\n")
            time.sleep(1)
            return chosen_model

        except Exception as e:
            self.console.print(f"\n[error]SYSTEM ERROR DURING SELECTION: {e}[/error]")
            return None

    
    def update_status(self, msg: str): self.status_msg = msg

    def log_thought(self, text: str):
        """Renders reasoning in a sophisticated, recessed panel."""
        if not text or not text.strip():
            return
        
        # Create a clean markdown version of the thought
        md = Markdown(text)
        
        # Render as a high-end 'Cognitive Block'
        panel = Panel(
            md,
            border_style="dim",
            box=box.SQUARE
        )
        self.console.print(panel)

    def log_action(self, tool: str, args: dict):
        """Renders the tool name prominently on its own line, followed by syntax below."""
        # 1. The Primary Command Line (Just EXECUTE > Tool Name)
        self.console.print(f"[dim]EXECUTE ❯[/dim] [dim]{tool}[/dim]")

        # 2. The Secondary Syntax Line (Indented and subtle)
        if args:
            arg_parts = []
            for k, v in args.items():
                # Formats strings with quotes, keeps numbers/bools raw
                val = f"'{v}'" if isinstance(v, str) else str(v)
                arg_parts.append(f"{k}={val}")
            
            syntax_str = f"({', '.join(arg_parts)})"
            # We use a subtle color and indentation to show it belongs to the line above
            self.console.print(f"  [#dim]{syntax_str}[/#dim]")

    def log_observation(self, text: str):
        """
        THE UNIVERSAL RENDERER.
        Automatically detects and renders Syntax, Markdown, or Plain Text.
        """
        # 1. Check for Code Tag (Priority 1)
        match = re.search(r"\[LANG:(\w+)\]\n([\s\S]*)", text)
        if match:
            lang = match.group(1)
            code_content = match.group(2).strip()
            self.console.print(f"  [subtle]↳[/subtle] [primary]CODE BLOCK ({lang.upper()}):[/primary]")
            syntax = Syntax(code_content, lang, theme="monokai", line_numbers=True, word_wrap=True)
            self.console.print(Panel(syntax, border_style="dim", box=box.SQUARE))
            return

        # 2. Check for Markdown (Priority 2: Tables, Headers, Lists)
        # We look for common markdown indicators like '#' or '|' or '-'
        if any(indicator in text for indicator in ['# ', '|', '---', '- [ ]', '* ']):
            self.console.print(f"  [subtle]↳[/subtle] [primary]STRUCTURED DATA:[/primary]")
            md = Markdown(text)
            self.console.print(Panel(md, border_style="dim", box=box.SQUARE))
            return

        # 3. Fallback to Plain Text (Priority 3)
        self.console.print(f"  [subtle]↳[/subtle] [subtle]{text}[/subtle]")

    def log_briefing(self, title: str, content: str):
        self.console.print(f"[primary]{title}[/primary][#111111 italic]{content}[/#111111 italic]\n")

    def log_final(self, text: str):
        if not text or not text.strip():
            return
        
        md = Markdown(text)
        # The 'Final Synthesis' is integrated into the Panel header with the blue highlight
        panel = Panel(
            md,
            border_style="green",
            box=box.ROUNDED,
            expand=True
        )
        self.console.print(panel)

    def log_error(self, text: str): self.console.print(f"\n[red italic][!] ERROR: {text}[/red italic]")

    def ask_user(self, question: str) -> str:
        self.console.print(f"\n[error][!] INTERVENTION REQUIRED:[/error] [white]{question}[/white]")
        self.console.print("[primary]❯ [/primary]", end="")
        sys.stdout.flush()
        return input()

    def log_system_status(self, msg: str): self.console.print(f"[dim italic][SYSTEM]: {msg}[/dim italic]")

    def log_mission_success(self): 
        self.console.print("[primary italic]ACHIEVED[/primary italic]")

# --- 2. THE TOOLBOX ---

class Toolbox:
    @staticmethod
    def list_files(directory=".", **kwargs):
        """Performs a deep scan of the target directory, returning detailed file metadata."""
        try:
            target = directory or kwargs.get('path') or kwargs.get('dir') or "."
            if not os.path.exists(target): return f"ERROR: Path '{target}' does not exist."
            
            items = os.listdir(target)
            report = []
            for item in items:
                full_path = os.path.join(target, item)
                stats = os.stat(full_path)
                is_dir = os.path.isdir(full_path)
                size = stats.st_size
                mod_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"[{'DIR' if is_dir else 'FILE'}] {item} | Size: {size}B | Modified: {mod_time}")
            
            return f"SUCCESS: Found {len(items)} items in '{target}':\n" + "\n".join(report)
        except Exception as e: return f"ERROR: Directory Scan Failed: {e}"

    @staticmethod
    def get_tree(directory=".", max_depth=3, **kwargs):
        """Generates a visual structural representation of the filesystem with depth control."""
        def _build_tree(path, prefix="", depth=0):
            if depth > max_depth: return ""
            tree = ""
            try:
                items = sorted([i for i in os.listdir(path) if not i.startswith('.')])
                for i, item in enumerate(items):
                    is_last = (i == len(items) - 1)
                    connector = "└── " if is_last else "├── "
                    tree += f"{prefix}{connector}{item}\n"
                    full_path = os.path.join(path, item)
                    if os.path.isdir(full_path):
                        extension = "    " if is_last else "│   "
                        tree += _build_tree(full_path, prefix + extension, depth + 1)
                return tree
            except Exception: return ""
        try: 
            target = directory or kwargs.get('path') or "."
            return f"SUCCESS: Project Structure (Max Depth {max_depth}):\n{_build_tree(target)}"
        except Exception as e: return f"ERROR: Tree Generation Failed: {e}"

    @staticmethod
    def read_file(filename=None, start_line=1, end_line=None, **kwargs):
        """Surgically reads file content. No truncation to prevent hallucinations."""
        try:
            target = filename or kwargs.get('path') or kwargs.get('file')
            if not target: return "ERROR: Missing filename."
            
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            for enc in encodings:
                try:
                    with open(target, "r", encoding=enc) as f:
                        lines = f.readlines()
                    content = lines
                    break
                except (UnicodeDecodeError, LookupError): continue
            
            if content is None: return "ERROR: Unable to decode file."

            s_idx = max(0, start_line - 1)
            e_idx = end_line if end_line else len(content)
            selected_lines = content[s_idx:e_idx]
            
            output = "".join(selected_lines).strip()
            ext = os.path.splitext(target)[1].lstrip('.') or 'text'
            return f"SUCCESS: [LANG:{ext}]\n{output}" 
        except Exception as e: return f"ERROR: Read Failure: {e}"

    @staticmethod
    def read_pdf(filename=None, **kwargs):
        """Extracts all text content from a PDF file."""
        try:
            target = filename or kwargs.get('path') or kwargs.get('file')
            if not target: return "ERROR: Missing filename."
            
            from pypdf import PdfReader
            reader = PdfReader(target)
            full_text = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text.append(f"--- Page {i+1} ---\n{text}")
            
            output = "\n".join(full_text)
            return f"SUCCESS: Extracted {len(reader.pages)} pages from '{target}':\n{output[:100000000000]}"
        except Exception as e: 
            return f"ERROR: PDF Read Failure: {e}"


    @staticmethod
    def write_file(filename=None, content=None, mode="overwrite", **kwargs):
        """Writes file and returns the FULL written content so agent can verify."""
        try:
            target = filename or kwargs.get('path') or kwargs.get('file')
            body = content or kwargs.get('text') or kwargs.get('body')
            if not target or body is None: return "ERROR: Missing parameters."
            mode = mode or kwargs.get('mode', 'overwrite')
            ext = os.path.splitext(target)[1].lstrip('.') or 'text'

            with open(target, "w", encoding='utf-8') as f: 
                f.write(str(body))
            return f"SUCCESS: File '{target}' written via {mode} mode.\n[LANG:{ext}]\n{str(body)}"
        except Exception as e: return f"ERROR: Write Failure: {e}"

    @staticmethod
    def append_to_file(filename=None, content=None, **kwargs):
        """Appends content to the end of a file with automatic newline handling."""
        try:
            target = filename or kwargs.get('path') or kwargs.get('file')
            body = content or kwargs.get('text') or kwargs.get('body')
            if not target or body is None: return "ERROR: Missing parameters."
            with open(target, "a", encoding='utf-8') as f: 
                f.write(f"\n{str(body)}")
            return f"SUCCESS: Content appended to '{target}'."
        except Exception as e: return f"ERROR: Append Failure: {e}"

    @staticmethod
    def replace_in_file(filename=None, search_pattern=None, replacement=None, use_regex=False, **kwargs):
        """Advanced text replacement. Can use standard strings or Regex patterns for complex edits."""
        try:
            target = filename or kwargs.get('path') or kwargs.get('file')
            search = search_pattern or kwargs.get('search')
            rep = replacement or kwargs.get('replace')
            if not target or not search or rep is None: return "ERROR: Missing parameters."
            
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                data = f.read()
            
            if use_regex:
                import re
                new_data = re.sub(search, rep, data)
            else:
                new_data = data.replace(search, rep)
            
            if data == new_data: return "ERROR: No matches found for replacement."
                
            with open(target, 'w', encoding='utf-8') as f:
                f.write(new_data)
            return f"SUCCESS: Transformation complete in {target}."
        except Exception as e: return f"ERROR: Replace Failure: {e}"


    @staticmethod
    def replace_lines(filename=None, start_line=None, end_line=None, content=None, **kwargs):
        """
        SURGICAL EDIT: Replaces a specific range of lines with new content.
        This is the safest way to edit large files without rewriting everything.
        """
        try:
            target = filename or kwargs.get('path') or kwargs.get('file')
            if not target or start_line is None or end_line is None or content is None:
                return "ERROR: Missing parameters (filename, start_line, end_line, and content required)."

            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Adjust for 0-based indexing
            start_idx = max(0, start_line - 1)
            end_idx = end_line # end_line is inclusive in human terms, exclusive in slicing

            if start_idx >= len(lines):
                return f"ERROR: Start line {start_line} is beyond file length ({len(lines)} lines)."

            # Prepare new content as a list of lines
            new_content_lines = [line + '\n' for line in str(content).splitlines()]
            if not new_content_lines[-1].endswith('\n'):
                new_content_lines[-1] += '\n'

            # Perform the surgical slice
            # We replace the segment from start_idx to end_idx with our new lines
            lines[start_idx:end_idx] = new_content_lines

            with open(target, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            return f"SUCCESS: Replaced lines {start_line}-{end_line} in '{target}'."
        except Exception as e:
            return f"ERROR: Line Replacement Failed: {e}"

    @staticmethod
    def patch_regex(filename=None, pattern=None, replacement=None, **kwargs):
        """
        PATTERN EDIT: Uses Regex to find and replace patterns. 
        Ideal for HTML tags, code syntax, or Markdown structural changes.
        """
        try:
            target = filename or kwargs.get('path') or kwargs.get('file')
            if not target or not pattern or replacement is None:
                return "ERROR: Missing parameters (filename, pattern, and replacement required)."

            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                data = f.read()

            import re
            new_data, count = re.subn(pattern, str(replacement), data)

            if count == 0:
                return f"ERROR: No matches found for pattern '{pattern}'."

            with open(target, 'w', encoding='utf-8') as f:
                f.write(new_data)

            return f"SUCCESS: Applied regex patch to '{target}'. Matches replaced: {count}"
        except Exception as e:
            return f"ERROR: Regex Patch Failed: {e}"

    @staticmethod
    def get_file_context(filename=None, line_start=1, line_end=None, **kwargs):
        """
        CONTEXT SCANNER: Reads a specific range of lines to help the agent 
        identify exactly which lines it needs to replace.
        """
        try:
            target = filename or kwargs.get('path') or kwargs.get('file')
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            s_idx = max(0, line_start - 1)
            e_idx = line_end if line_end else len(lines)
            selected = lines[s_idx:e_idx]
            
            output = []
            for i, line in enumerate(selected):
                output.append(f"{s_idx + i + 1}: {line}")
            
            return f"SUCCESS: Context for '{target}' (Lines {line_start}-{e_idx}):\n" + "\n".join(output)
        except Exception as e:
            return f"ERROR: Context Scan Failed: {e}"

    # --- CODE INTELLIGENCE & ANALYSIS ---

    @staticmethod
    def analyze_code_structure(filename=None, **kwargs):
        try:
            target = filename or kwargs.get('path') or kwargs.get('file')
            with open(target, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            structure = {"classes": [], "functions": [], "imports": []}
            for node in tree.body: # Iterate only top-level nodes
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    structure["classes"].append({"name": node.name, "methods": methods})
                elif isinstance(node, ast.FunctionDef):
                    structure["functions"].append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    structure["imports"].append(ast.dump(node))

            return f"SUCCESS: Logical Map of {target}:\n{json.dumps(structure, indent=2)}"
        except Exception as e: return f"ERROR: AST Analysis Failed: {e}"

    @staticmethod
    def python_executor(code=None, **kwargs):
        """Executes code and returns the FULL source + output to prevent logic gaps."""
        import os, sys, subprocess, tempfile, traceback, re
        clean_code = code or kwargs.get('script') or kwargs.get('command') or kwargs.get('code')
        if not clean_code: return "ERROR: No valid code detected."

        blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", clean_code, re.DOTALL)
        if blocks: clean_code = blocks[-1].strip()

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as tmp:
                tmp.write(clean_code)
                tmp_path = tmp.name

            result = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=30)
            
            # Return the full code snippet used so the LLM can verify its own logic
            payload = f"[LANG:python]\n{clean_code}\n\n"
            if result.returncode != 0:
                payload += f"--- EXECUTION ERROR ---\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            else:
                payload += f"--- EXECUTION SUCCESS ---\nSTDOUT: {result.stdout}\n{f'STDERR: {result.stderr}' if result.stderr else ''}"
            return payload
        except Exception as e: return f"ERROR: Engine Failure: {traceback.format_exc()}"
        finally:
            if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

    # --- SYSTEM & NETWORK ORCHESTRATION ---

    @staticmethod
    def get_system_info():
        """Returns a comprehensive snapshot of the host environment."""
        import platform, psutil # Note: Requires psutil if installed, else falls back to platform
        try:
            info = {
                "os": f"{platform.system()} {platform.release()}",
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": sys.version,
                "cpu_count": os.cpu_count(),
                "cwd": os.getcwd(),
                "user": os.getlogin() if os.name != 'nt' else os.environ.get('USERNAME')
            }
            # Try adding memory/disk via psutil if available
            try:
                import psutil
                info["memory_total"] = f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB"
                info["disk_usage"] = f"{psutil.disk_usage('/').percent}%"
            except ImportError: pass
            return f"SUCCESS: System Manifest:\n{json.dumps(info, indent=2)}"
        except Exception as e: return f"ERROR: System Info Failed: {e}"

    @staticmethod
    def shell_execute(command):
        """Executes low-level system commands with full terminal capability."""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=45)
            return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        except Exception as e: return f"ERROR: Shell Execution Failed: {e}"

    @staticmethod
    def manage_process(action, pid=None, **kwargs):
        """Orchestrates system processes (list, kill). action='list' or 'kill'."""
        try:
            if action == "list":
                import subprocess
                cmd = "tasklist" if os.name == 'nt' else "ps aux"
                res = subprocess.check_output(cmd, shell=True, text=True)
                return f"SUCCESS: Process List:\n{res[:5000]}"
            elif action == "kill":
                if not pid: return "ERROR: PID required for kill action."
                import os, signal
                if os.name == 'nt':
                    os.system(f"taskkill /F /PID {pid}")
                else:
                    os.kill(int(pid), signal.SIGKILL)
                return f"SUCCESS: Process {pid} terminated."
            return "ERROR: Invalid action. Use 'list' or 'kill'."
        except Exception as e: return f"ERROR: Process Management Failed: {e}"

    # --- MATHEMATICAL & DATA COMPUTATION ---

    @staticmethod
    def scientific_compute(expression):
        """Performs high-precision symbolic, arithmetic, and matrix computation using math and numpy."""
        import math
        import numpy as np
        from decimal import Decimal, getcontext
        getcontext().prec = 60 
        safe_namespace = {
            "math": math, 
            "np": np,             # Provides access to all numpy functions via 'np.'
            "Decimal": Decimal, 
            "pi": math.pi, 
            "e": math.e, 
            "sin": math.sin, 
            "cos": math.cos, 
            "tan": math.tan, 
            "sqrt": math.sqrt,
            "log": math.log, 
            "exp": math.exp
        }
        try:
            # The agent can now use expressions like "np.mean([1, 2, 3])" or "np.array([1,2]) * 2"
            result = eval(expression, {"__builtins__": None}, safe_namespace)
            return f"SUCCESS: Computed Result = {result}"
        except Exception as e: return f"ERROR: Math Engine Error: {e}"

    @staticmethod
    def search_files(pattern, **kwargs):
        """Finds files matching a glob pattern recursively."""
        try:
            target = kwargs.get('directory', '.')
            matches = glob.glob(f"{target}/**/{pattern}", recursive=True)
            return f"SUCCESS: Found {len(matches)} matches:\n" + "\n".join(matches)
        except Exception as e: return f"ERROR: Search Failed: {e}"

    @staticmethod
    def grep_search(pattern, file_path, context_lines=3, **kwargs):
        """Searches for a pattern within a file and returns the matching lines with surrounding context."""
        try:
            target = file_path or kwargs.get('path') or kwargs.get('file')
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            results = []
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = lines[start:end]
                    results.append(f"--- Match at Line {i+1} ---\n" + "".join(context))
            
            if not results: return "SUCCESS: No matches found."
            return f"SUCCESS: Found {len(results)} match blocks:\n" + "\n".join(results[:5])
        except Exception as e: return f"ERROR: Grep Failed: {e}"
    
    @staticmethod
    def run_file(filename):
        """Executes a specific python file using the current system interpreter."""
        try:
            import sys, subprocess
            # We use sys.executable to ensure we use the same environment the agent is running in
            result = subprocess.run([sys.executable, filename], capture_output=True, text=True, timeout=30)
            return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        except Exception as e: 
            return f"ERROR: Failed to execute file '{filename}': {e}"

    @staticmethod
    def ask_user(question):
        """Directly interrupts the autonomous loop to request human intervention."""
        print(f"\n{'\033[93m'}[!] AGENT INTERVENTION REQUIRED: {question}\033[0m")
        return f"USER RESPONSE: {input('❯ ')}"

    @staticmethod
    def get_file_stats(filename=None, **kwargs):
        """Provides deep-dive metadata for a specific file (permissions, size, timestamps)."""
        try:
            target = filename or kwargs.get('path') or kwargs.get('file')
            s = os.stat(target)
            return f"SUCCESS: Stats for {target}:\nSize: {s.st_size}B\nCreated: {datetime.fromtimestamp(s.st_ctime)}\nModified: {datetime.fromtimestamp(s.st_mtime)}\nMode: {oct(s.st_mode)}"
        except Exception as e: return f"ERROR: Stats Failed: {e}"

class ToolCompilerError(Exception):
    """Custom exception for failed compilation stages."""
    pass


TOOL_MAP = {
    "list_files": Toolbox.list_files, 
    "get_tree": Toolbox.get_tree,
    "search_files": Toolbox.search_files,
    "grep_search": Toolbox.grep_search,
    "get_system_info": Toolbox.get_system_info,
    "read_file": Toolbox.read_file, 
    "write_file": Toolbox.write_file, 
    "append_to_file": Toolbox.append_to_file,
    "replace_in_file": Toolbox.replace_in_file,
    "get_file_stats": Toolbox.get_file_stats,
    "python_executor": Toolbox.python_executor,
    "run_file": Toolbox.run_file, # Note: run_file is a wrapper for shell_execute/python_executor in logic
    "shell_execute": Toolbox.shell_execute,
    "manage_process": Toolbox.manage_process,
    "ask_user": Toolbox.ask_user,
    "scientific_compute": Toolbox.scientific_compute,
    "read_pdf": Toolbox.read_pdf,
    "replace_lines": Toolbox.replace_lines,
    "patch_regex": Toolbox.patch_regex,
    "get_file_context": Toolbox.get_file_context
}

class LangChainToolManager:
    """
    Wraps your existing Toolbox into a standardized format.
    This allows us to move away from the custom 'ToolCompiler'.
    """
    def __init__(self, toolbox_class):
        self.toolbox = toolbox_class
        # This map links the string name (from LLM) to the actual function
        self.registry = {
            "list_files": Toolbox.list_files, 
            "get_tree": Toolbox.get_tree,
            "search_files": Toolbox.search_files,
            "grep_search": Toolbox.grep_search,
            "get_system_info": Toolbox.get_system_info,
            "read_file": Toolbox.read_file, 
            "write_file": Toolbox.write_file, 
            "append_to_file": Toolbox.append_to_file,
            "replace_in_file": Toolbox.replace_in_file,
            "get_file_stats": Toolbox.get_file_stats,
            "python_executor": Toolbox.python_executor,
            "run_file": Toolbox.run_file, # Note: run_file is a wrapper for shell_execute/python_executor in logic
            "shell_execute": Toolbox.shell_execute,
            "manage_process": Toolbox.manage_process,
            "ask_user": Toolbox.ask_user,
            "scientific_compute": Toolbox.scientific_compute,
            "read_pdf": Toolbox.read_pdf,
            "replace_lines": Toolbox.replace_lines,
            "patch_regex": Toolbox.patch_regex,
            "get_file_context": Toolbox.get_file_context
        }


    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name not in self.registry:
            return f"ERROR: Tool '{tool_name}' not found."
        
        try:
            func = self.registry[tool_name]
            # PATCH: If args is a string (common LLM error), attempt to parse it as JSON.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return f"ERROR: Tool arguments provided as string but are not valid JSON: {args}"
            
            result = func(**args)
            return str(result)
        except Exception as e:
            return f"ERROR: Execution failed: {str(e)}"

class LLMInterface:
    def __init__(self, model_name: str, base_system_prompt: str):
        self.llm = ChatOllama(model=model_name, temperature=0.8)
        self.base_system_prompt = base_system_prompt
        self.parser = PydanticOutputParser(pydantic_object=AgentResponse)

    def generate_response_streamed(self, history: List[Dict[str, str]], user_input: str, ui: Any, status_callback=None) -> 'AgentResponse':
        full_system_prompt = self.base_system_prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", full_system_prompt),
            ("system", "Format your response as a JSON object following these instructions:\n{format_instructions}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        formatted_prompt = prompt.format_messages(
            format_instructions=self.parser.get_format_instructions(),
            history=history,
            input=user_input
        )

        full_content = ""
        try:
            for chunk in self.llm.stream(formatted_prompt):
                content = chunk.content
                if content:
                    full_content += content
                    
                    # --- THE MINIMALIST STATUS SNIFFER ---
                    if status_callback:
                        # We look for the "status_summary" key specifically.
                        # This regex captures the value of status_summary even if the JSON is incomplete.
                        status_match = re.search(r'"status_summary":\s*"([^"]*)', full_content)
                        
                        if status_match:
                            clean_status = status_match.group(1).replace('\\"', '"').strip()
                            if clean_status:
                                # We only update the UI if there's actual text to show
                                status_callback(clean_status)

            
            def attempt_parse(raw_text: str) -> AgentResponse:
                cleaned = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', raw_text, flags=re.DOTALL)
                start_idx = cleaned.find('{')
                end_idx = cleaned.rfind('}')
                if start_idx == -1 or end_idx == -1:
                    raise ValueError("No JSON braces found.")
                json_candidate = cleaned[start_idx:end_idx + 1]
                try:
                    return self.parser.parse(json_candidate)
                except Exception:
                    repaired = re.sub(r'(?<!\\)\n', '\\n', json_candidate)
                    repaired = repaired.replace('\\\\n', '\\n')
                    return self.parser.parse(repaired)
                pass

            try:
                return self.parser.parse(full_content)
            except Exception:
                return attempt_parse(full_content)

        except Exception as e:
            raise ValueError(f"CRITICAL PARSE FAILURE | Error: {str(e)}")     


# --- PATCH 4: THE GRAPH ORCHESTRATOR ---

class LangChainEngine:
    def __init__(self, model: str, ui: Any, toolbox_manager: LangChainToolManager):
        self.ui = ui
        self.tool_manager = toolbox_manager
        self.llm_interface = LLMInterface(model, self._get_system_prompt())

        self.graph = self._build_graph()
        
        self.initial_state: AgentState = {
            "mission": "",
            "trajectory": "",
            "synthesis": "",
            "last_observation": "",
            "history": [],
            "next_step": "plan",
            "current_node": "init",
            "error_count": 0,
            "is_finished": False,
            "pending_action": None,
            "consecutive_reasoning_steps": 0,
        }

    def _get_system_prompt(self) -> str:
        """MOON"""
        return """
        [IDENTITY: MOON]
        Maintain precision and politeness in every thought and action. You're an Autonomous Agent. 

        [OFFLINE]
        - Rely on internal weights for knowledge.
        - Use tools when it's necessary.

        [TOOLKIT]
        - list_files(directory='.')
        - get_tree(directory='.', max_depth=3)
        - search_files(pattern, directory='.')
        - grep_search(pattern, file_path, context_lines=3)
        - python_executor(code)
        - run_file(filename)
        - shell_execute(command)
        - manage_process(action, pid=None)
        - get_system_info()
        - get_file_stats(filename)
        - scientific_compute(expression) (Supports all numpy functions via 'np., e.g., np.sin(1))
        - read_file(filename, start_line=1, end_line=None)
        - read_pdf (filename)
        - write_file(filename, content, mode='overwrite')
        - append_to_file(filename, content)
        - replace_in_file(filename, search_pattern, replacement)
        - get_file_context(filename, line_start, line_end)
        - replace_lines(filename, start_line, end_line, content)
        - patch_regex(filename, pattern, replacement)
        
        [OUTPUT PROTOCOL]
        You respond with a valid JSON object containing these exact keys:

        1. "trajectory": "Step X/Y | Goal"
        2. "status_summary": active status (e.g., 'Scanning files').
        3. "synthesis": Thinking.
        4. "action": (object or null) If an action is required, provide an object with:
           {{"tool_name": "name_of_tool", "arguments": {{"arg_name": "value"}}}}
           If no action is required, set this to null.
        5. "completion_summary": "Final report" or null
        """


    # --- THE NODES (Refined for LangGraph) ---
    # LangGraph nodes take 'state' and return 'updates' to that state.

    def node_llm(self, state: AgentState) -> Dict[str, Any]:
        """
        NODE: LLM (Cognitive Core).
        UPDATED: Now passes a status callback for real-time dynamic updates.
        """
        with self.ui.node_context("AI") as live:
            try:
                live.update("Decision") 
                
                obs = state.get('last_observation', '')
                input_context = f"MISSION: {state['mission']}\nLAST OBSERVATION: {obs}"
                
                response: AgentResponse = self.llm_interface.generate_response_streamed(
                    state["history"], 
                    input_context, 
                    self.ui,
                    status_callback=live.update # The UI now receives the clean 'status_summary'
                )
                
                self.ui.log_thought(response.synthesis)

                action_text = f" | ACTION: {response.action.tool_name}" if response.action else ""
                clean_assistant_message = f"THOUGHT: {response.synthesis}{action_text}"

                # Logic for managing reasoning loops and state transitions
                current_reasoning_count = state.get("consecutive_reasoning_steps", 0)
                if response.action or response.completion_summary:
                    next_step = "plan" if not response.completion_summary else "complete"
                    new_reasoning_count = 0
                else:
                    current_reasoning_count += 1
                    new_reasoning_count = current_reasoning_count
                    next_step = "human" if current_reasoning_count >= 3 else "plan"

                new_history = list(state["history"])
                new_history.append({"role": "assistant", "content": clean_assistant_message})

                updates = {
                    "current_node": "llm",
                    "trajectory": response.trajectory,
                    "synthesis": response.synthesis,
                    "consecutive_reasoning_steps": new_reasoning_count,
                    "history": new_history
                }

                if response.completion_summary:
                    updates["is_finished"] = True
                    updates["next_step"] = "complete" 
                    updates["last_observation"] = f"# MISSION REPORT\n{response.completion_summary}"
                elif response.action:
                    updates["next_step"] = "act"
                    updates["pending_action"] = response.action
                else:
                    updates["next_step"] = next_step

                return updates

            except Exception as e:
                # The context manager handles the visual 'FAILED' status via its internal try/except
                return {
                    "current_node": "error",
                    "next_step": "error",
                    "last_observation": f"LLM_FAILURE: {str(e)}",
                    "error_count": state["error_count"] + 1,
                    "consecutive_reasoning_steps": 0 
                }


    def node_action(self, state: AgentState) -> Dict[str, Any]:
        """
        NODE: KINETIC. Executes tools and shows a minimalist status update.
        REDESIGNED: Removed the heavy 'Success Box' in favor of streamlined status updates.
        """
        with self.ui.node_context("TOOL") as live:
            action = state.get("pending_action")
            if not action: return {"next_step": "plan"}

            tool_name = action.tool_name
            args = action.arguments
            
            # 1. Update the single-line status to show active execution
            live.update(f"Using {tool_name}...") 
            
            # 2. Log the action details to the main terminal scroll (the history)
            self.ui.log_action(tool_name, args)
            
            # 3. Execute the tool
            observation = self.tool_manager.execute(tool_name, args)
            
            # Handle error detection within the observation string
            is_error = "ERROR" in str(observation).upper()
            if is_error:
                observation = f"!!! TOOL EXECUTION FAILURE !!!\n{observation}"

            # Prevent memory overflow with massive outputs
            MAX_OBSERVATION_LENGTH = 10000000 # Reduced from your original huge number for safety
            if len(str(observation)) > MAX_OBSERVATION_LENGTH:
                observation = str(observation)[:MAX_OBSERVATION_LENGTH] + "\n\n[TRUNCATED]"

            # 4. Log the observation to the main terminal scroll (the history)
            self.ui.log_observation(observation)

            # 5. Update the single-line status one last time before the context closes
            if is_error:
                live.update(f"{tool_name} FAILED ✕")
            else:
                live.update(f"{tool_name} Done.")

            return {
                "current_node": "action",
                "last_observation": observation,
                "history": state["history"] + [{"role": "user", "content": f"Observation: {observation}"}],
                "pending_action": None, 
                "next_step": "plan",
                "consecutive_reasoning_steps": 0 
            }

    def node_human(self, state: AgentState) -> Dict[str, Any]:
        """
        NODE: HUMAN (Intervention).
        REDESIGNED: Minimalist status line; maintains standard interaction.
        """
        with self.ui.node_context("NODE") as live:
            live.update("Awaiting strategic directive...")
            user_input = self.ui.ask_user("Provide strategic guidance:")
            return {
                "current_node": "human",
                "last_observation": f"USER_DIRECTIVE: {user_input}",
                "history": state["history"] + [{"role": "user", "content": user_input}],
                "next_step": "plan"
            }

    def node_complete(self, state: AgentState) -> Dict[str, Any]:
        """
        NODE: TERMINAL (High-Fidelity Synthesis).
        Renders the final result as a polished document piece.
        """
        with self.ui.node_context("REPORT") as live:
            live.update("Synthesizing results...")
            raw_text = state.get("last_observation", "")
            
            # 1. Clean the text: Remove common LLM-injected headers like '# MISSION REPORT'
            clean_text = re.sub(r'^#\s*MISSION\s*REPORT\s*', '', raw_text, flags=re.MULTILINE | re.IGNORECASE)
            clean_text = re.sub(r'^\[REPORT\]\s*', '', clean_text, flags=re.IGNORECASE)

            # 2. Render the high-end synthesis
            if clean_text.strip():
                # Use Markdown for professional formatting (tables, bold, etc)
                self.ui.log_final(clean_text)
            else:
                self.ui.console.print("[dim]No summary available.[/dim]")

        return {
            "is_finished": True,
            "next_step": "end" 
        }

    def node_error(self, state: AgentState) -> Dict[str, Any]:
        """
        NODE: MEDIC (Error Recovery).
        REDESIGNED: Removed the massive Red Box. Uses high-impact inline logging 
        and a status line to signal failure without breaking terminal flow.
        """
        with self.ui.node_context("ERROR", is_error=True) as live:
            live.update("Analyzing error")
            new_error_count = state["error_count"] + 1
            err_msg = str(state['last_observation'])[:200]
            
            # Log the error to the main stream with high visibility
            self.ui.log_error(f"CRITICAL: {err_msg}")

            if new_error_count >= 3:
                live.update("TERMINATING SYSTEM")
                return {
                    "current_node": "error",
                    "next_step": "terminate",
                    "error_count": new_error_count
                }
            
            error_context = (
                f"CRITICAL SYSTEM ERROR:\n{state['last_observation']}\n\n"
                f"INSTRUCTION: Analyze the failure above. Do not repeat the same action. "
                f"Pivot your strategy immediately."
            )
            
            return {
                "current_node": "error",
                "next_step": "plan",
                "error_count": new_error_count,
                "last_observation": error_context,
                "history": state["history"] + [{"role": "user", "content": error_context}]
            }
    # --- THE ROUTER (The Conditional Logic) ---

    def conditional_router(self, state: AgentState):
        """The 'brain' that decides which edge to follow."""
        step = state.get("next_step", "llm")
        
        valid_steps = ["plan", "act", "human", "error", "complete", "terminate"]
        if step not in valid_steps:
            print(f"\n{self.ui.BROWN}[!] WARNING: Invalid routing step '{step}'. Defaulting to LLM.{self.ui.RESET}")
            return "llm"
            
        return step

    # --- THE GRAPH CONSTRUCTION ---

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("llm", self.node_llm)
        workflow.add_node("action", self.node_action)
        workflow.add_node("human", self.node_human)
        workflow.add_node("error", self.node_error)
        workflow.add_node("complete", self.node_complete) 

        workflow.add_edge(START, "llm")

        def router(state: AgentState):
            return state["next_step"]

        workflow.add_conditional_edges(
            "llm",
            router,
            {
                "plan": "llm",
                "act": "action",
                "human": "human",
                "error": "error",
                "complete": "complete", 
                "terminate": END
            }
        )

        workflow.add_edge("action", "llm")
        workflow.add_edge("human", "llm")
        workflow.add_edge("error", "llm")
        workflow.add_edge("complete", END)

        return workflow.compile()

    def run_mission(self, mission: str, previous_state: Optional[Dict[str, Any]] = None):
        # 1. Wipe the screen of any 'Loading weights' or old terminal junk
        self.ui._force_clear()
        
        self.ui.log_briefing("", mission)

        if previous_state:
            self.ui.log_system_status("Resuming existing cognitive session")
            inputs = {**previous_state, "mission": mission}
            inputs["is_finished"] = False
            inputs["next_step"] = "plan"
        else:
            inputs = {
                **self.initial_state,
                "mission": mission,
                "history": [{"role": "system", "content": self.llm_interface.base_system_prompt}],
            }

        final_state = self.graph.invoke(inputs)
        return final_state

# --- 4. MAIN BOOTSTRAPPER ---

if __name__ == "__main__":
    ui = RichUI()
    
    try:
        ui.startup_sequence()
        
        # 1. INITIALIZATION PHASE
        selected_model = ui.select_model()
        if not selected_model:
            ui.log_error("No model selected. Exiting.")
            sys.exit()

        toolbox = Toolbox()
        tool_manager = LangChainToolManager(toolbox)
        engine = LangChainEngine(selected_model, ui, tool_manager)
        
        ui.boot()

        # 2. CONTINUOUS MISSION LOOP
        last_known_state = None 

        while True:
            # A clean, cinematic prompt for the next instruction
            ui.console.print(f"\n[primary]❯❯[/primary] ", end="")
            sys.stdout.flush()
            goal = input().strip()
            
            if goal.lower() in ['exit', 'quit', 'q']:
                break
                
            if goal:
                # Execute the mission
                final_state = engine.run_mission(goal, previous_state=last_known_state)
                
                # Check if we should keep context or wipe it for a fresh start
                # If the user wants to continue, we save state; otherwise, we reset.
                # We allow the user to type 'reset' to clear memory.
                if goal.lower() == 'reset':
                    last_known_state = None
                    engine = LangChainEngine(selected_model, ui, tool_manager)
                    ui.log_system_status("Cognitive memory purged.")
                else:
                    last_known_state = final_state 
            else:
                ui.log_system_status("No input detected.")


    except KeyboardInterrupt:
        ui.console.print("\n\n[error][SYSTEM]: INTERRUPT DETECTED. SHUTTING DOWN...[/error]")
    except Exception as e:
        ui.log_error(f"System Crash: {str(e)}")
        traceback.print_exc()
    finally:
        ui.console.print(f"\n[dim][SYSTEM]: SESSION CLOSED.[/dim]")