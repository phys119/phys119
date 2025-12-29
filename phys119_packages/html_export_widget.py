import ipywidgets as widgets

# This variable will hold the widget instance once it’s created.
export_toggle = None
status_label = None
ui = None


def get_export_ui():
    """
    Returns a HBox containing the toggle button and status label.
    Creates them on first call; subsequent calls just return the existing UI.
    """
    global export_toggle, status_label, ui

    # If the widget already exists, just return the UI.
    if export_toggle is not None:
        return ui, export_toggle

    # First time we run – build everything.
    export_toggle = widgets.ToggleButton(
        value=False,
        description='HTML rendering',
        tooltip='Turn on to generate HTML rendering for export',
        button_style='info'   # blue / gray background
    )

    status_label = widgets.Label(value="ℹ️ HTML rendering is disabled.")

    def _on_toggle_change(change):
        """Callback triggered whenever `export_toggle.value` changes."""
        if change['new']:                     # button turned ON
            status_label.value = "✅ HTML rendering is enabled."
            export_toggle.button_style = 'success'   # green when active
        else:                                 # button turned OFF
            status_label.value = "ℹ️ HTML rendering is disabled."
            export_toggle.button_style = 'info'      # revert to original style

    export_toggle.observe(_on_toggle_change, names='value')

    ui = widgets.HBox([export_toggle, status_label])
    return ui, export_toggle