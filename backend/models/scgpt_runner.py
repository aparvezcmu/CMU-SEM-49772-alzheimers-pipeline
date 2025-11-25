"""
Simple SCGPT runner / fallback plot generator.

This module tries to use an installed `scgpt` package if available.
If it's not present, it falls back to a tiny parser that creates
representative matplotlib visualizations from simple prompts.

The main function `generate_plot_from_prompt(prompt)` returns a dict
with at least a `base64` key containing a PNG image encoded as base64.
"""
from typing import Dict
import io
import base64
import traceback

try:
    # If a specialized scgpt package exists, we'll try to use it.
    import scgpt  # type: ignore
    _HAS_SCGPT = True
except Exception:
    _HAS_SCGPT = False

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    buf.seek(0)
    b = buf.getvalue()
    return base64.b64encode(b).decode("utf-8")

def _fallback_plot(prompt: str) -> Dict:
    """Create a simple matplotlib visualization based on keywords in the prompt."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6,4))
    lower = prompt.lower()

    try:
        if 'hist' in lower or 'histogram' in lower:
            data = np.random.normal(loc=0.0, scale=1.0, size=500)
            ax.hist(data, bins=30, color='C0', alpha=0.8)
            ax.set_title('Histogram (random sample)')

        elif 'scatter' in lower:
            x = np.random.rand(100)
            y = np.random.rand(100)
            colors = np.random.rand(100)
            sizes = 50 * np.random.rand(100)
            ax.scatter(x, y, c=colors, s=sizes, cmap='viridis', alpha=0.8)
            ax.set_title('Scatter (random sample)')

        elif 'sine' in lower or 'sin' in lower:
            x = np.linspace(0, 10, 400)
            y = np.sin(x)
            ax.plot(x, y, '-r')
            ax.set_title('Sine wave')

        else:
            # Default: simple line plot (demo)
            x = np.linspace(0, 10, 200)
            y = np.sin(x) + 0.3 * np.random.randn(len(x))
            ax.plot(x, y, '-o', markersize=3, alpha=0.6)
            ax.set_title('Line plot (demo)')

        ax.grid(True)
        base64_img = _fig_to_base64(fig)
        plt.close(fig)
        return {"base64": base64_img, "meta": {"source": "fallback"}}

    except Exception as e:
        plt.close('all')
        return {"error": str(e), "traceback": traceback.format_exc()}

def generate_plot_from_prompt(prompt: str) -> Dict:
    """Generate a plot from a natural language prompt.

    Returns:
        dict: {"base64": "...", "meta": {...}} on success
              or {"error": ..., "traceback": ...} on failure
    """
    # First, if an scgpt package is available, prefer it
    if _HAS_SCGPT:
        try:
            # Attempt to call a hypothetical scgpt API. If scgpt
            # provides a different interface adjust accordingly.
            img_bytes = scgpt.generate_image_from_prompt(prompt)
            if isinstance(img_bytes, bytes):
                return {"base64": base64.b64encode(img_bytes).decode('utf-8'), "meta": {"source": "scgpt"}}
            # If scgpt returns a PIL Image
            try:
                from PIL import Image
                buf = io.BytesIO()
                img_bytes.save(buf, format='PNG')
                return {"base64": base64.b64encode(buf.getvalue()).decode('utf-8'), "meta": {"source": "scgpt"}}
            except Exception:
                pass

        except Exception as e:
            return {"error": f"scgpt failure: {e}", "traceback": traceback.format_exc()}
    # Next, try to run plotting code inside the Jupyter kernel so it can access
    # the kernel's current state (variables, data, imports).
    try:
        # Import kernel manager lazily to avoid circular imports when module is loaded
        from kernel_manager import get_kernel_manager

        km = get_kernel_manager()

        # Build Python code to run inside kernel. It constructs a matplotlib
        # figure according to the prompt, saves it to a BytesIO buffer, and
        # displays it as an IPython Image (this produces an 'image/png' display).
        safe_prompt = repr(prompt)
        code = f"""
prompt = {safe_prompt}
import io
from IPython.display import Image, display
# Use existing numpy/matplotlib imports available in the kernel
fig = None
try:
    fig = plt.figure(figsize=(6,4))
    lower = prompt.lower()
    if 'hist' in lower or 'histogram' in lower:
        data = np.random.normal(loc=0.0, scale=1.0, size=500)
        plt.hist(data, bins=30, color='C0', alpha=0.8)
        plt.title('Histogram (kernel)')
    elif 'scatter' in lower:
        x = np.random.rand(100)
        y = np.random.rand(100)
        colors = np.random.rand(100)
        sizes = 50 * np.random.rand(100)
        plt.scatter(x, y, c=colors, s=sizes, cmap='viridis', alpha=0.8)
        plt.title('Scatter (kernel)')
    elif 'sine' in lower or 'sin' in lower:
        x = np.linspace(0, 10, 400)
        y = np.sin(x)
        plt.plot(x, y, '-r')
        plt.title('Sine wave (kernel)')
    else:
        x = np.linspace(0, 10, 200)
        y = np.sin(x) + 0.3 * np.random.randn(len(x))
        plt.plot(x, y, '-o', markersize=3, alpha=0.6)
        plt.title('Line plot (kernel)')
    plt.grid(True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    display(Image(data=buf.getvalue()))
finally:
    try:
        if fig:
            plt.close(fig)
    except Exception:
        pass
"""

        exec_result = km.execute_code(code)

        # exec_result should contain display outputs; prefer 'image' outputs
        outputs = exec_result.get('outputs', [])
        for out in outputs:
            if out.get('type') == 'image' and out.get('format') == 'png':
                return {"base64": out.get('content'), "meta": {"source": "kernel"}}
            if out.get('type') == 'html' and isinstance(out.get('content'), str):
                # try to extract base64 from an <img src="data:image/png;base64,...">
                import re
                m = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', out.get('content'))
                if m:
                    return {"base64": m.group(1), "meta": {"source": "kernel", "format": "html-img"}}

        # If kernel didn't produce an image, fall back to local generator
    except Exception as e:
        # If kernel is not available or execution failed, we'll fall back below
        pass

    # Final fallback if scgpt and kernel execution both unavailable/failed
    return _fallback_plot(prompt)
