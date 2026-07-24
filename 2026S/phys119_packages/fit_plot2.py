import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from matplotlib.lines import Line2D
import os
import base64
import pickle
import gc
from IPython.display import clear_output

objs = []
names = []

par_form="{:.6g}" # format for parameters in boxes.
datsize=4.5 # size of data points on both graphs
ptsize=2.5 # size of markers that shown line "end points"

def get_file_name(fname):
    # Create a safe filename
    fname = base64.urlsafe_b64encode(bytes(fname, 'utf-8')).decode('utf-8')
    
    # Get the base directory for storing temporary files
    if 'JPY_SESSION_NAME' in os.environ:
        # Running in Jupyter
        base, name = os.path.split(os.environ['JPY_SESSION_NAME'])
        if len(name) > 6 and name.endswith('.ipynb'):
            name = name[:-6]
        name = '.' + name
    else:
        # Running in VS Code or other environment
        base = os.path.join(os.path.expanduser('~'), '.fit_plot_cache')
        name = 'vscode_session'
        # Create the cache directory if it doesn't exist
        os.makedirs(base, exist_ok=True)
    
    # Create a unique filename
    newname = name + '-' + fname + '.pkl'
    return os.path.join(base, newname)

class generic_fit_with_background:
    """Generic Class for creating fit objects

    Parameters:
    name: a unique name that is used as a plot title as well as for tagging the fit parameters
    xdata: x values of the data
    ydata: y values
    yerr: uncertainties in y values
    use_background: a boolean. If false we just do a linear fit, if true the function is
            np.log(np.exp(self.intercept)*np.exp(self.cxp*self.slope)+np.exp(self.yoff))
    input_boxes: a boolean. If True display boxes that allow manual setting of parameters
    """
    
    def cleanup_existing(self, name):
        """Clean up existing widget with the same name"""
        global objs, names
        try:
            # Find and remove any existing widget with the same name
            for i, obj_name in enumerate(names):
                if obj_name == name:
                    # Remove the widget from display
                    if hasattr(objs[i], 'out'):
                        objs[i].out.clear_output()
                    # Close the figure
                    if hasattr(objs[i], 'fig') and plt.fignum_exists(objs[i].fig.number):
                        plt.close(objs[i].fig)
                    # Remove from our tracking lists
                    objs.pop(i)
                    names.pop(i)
                    break
            # Force garbage collection
            gc.collect()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    def save_state(self):
        """Save the current state to a file"""
        try:
            with open(self.filename, 'wb') as ff:
                pickle.dump((self.xp, self.yp, self.yoff), ff)
        except Exception as e:
            print(f"Warning: Could not save state: {e}")
    
    def update_displays(self, initial_scale=False):
        """Update the graphs and other output
        
        Parameters:
        initial_scale: If True, calculate initial scaling based on data
        """
        try:
            # Store current view limits if this is not the initial scale
            if not initial_scale and hasattr(self, 'ax') and len(self.ax) > 1:
                xl = self.ax[1].get_xlim()
                yl = self.ax[1].get_ylim()
            else:
                # For initial scale, use data range with some padding
                x_padding = (np.max(self.xdata) - np.min(self.xdata)) * 0.1
                y_padding = (np.max(self.ydata) - np.min(self.ydata)) * 0.2
                xl = (np.min(self.xdata) - x_padding, 
                     np.max(self.xdata) + x_padding)
                yl = (min(np.min(self.ydata - self.yerr) - y_padding, 
                         np.min(self.ydata) * 0.9) if np.any(self.ydata < 0) 
                     else max(0, np.min(self.ydata - self.yerr) - y_padding),
                     max(np.max(self.ydata + self.yerr) + y_padding, 
                         np.max(self.ydata) * 1.1) if np.max(self.ydata) != 0 else 1.1)
            
            # Clear both subplots
            self.ax[0].clear()
            self.ax[1].clear()
            
            # Plot main data and fit
            self.ax[1].errorbar(self.xdata, self.ydata, self.yerr, 
                              fmt='ro', label="Data", markersize=datsize)
            
            # Create fit line with proper styling
            self.line = Line2D(self.cxp, self.cyp, color='b', linestyle='-', label="Fit")
            self.ax[1].add_line(self.line)
            
            # Create data points with proper styling
            self.line2 = Line2D(self.xp, self.yp, marker='o', color='b', 
                              linestyle='', markersize=ptsize, label="Anchor Points")
            self.ax[1].add_line(self.line2)
            self.ax[1].set_title(self.name if hasattr(self, 'name') else 'Fit Plot')
            self.ax[1].legend()
            
            # Plot residuals
            self.ax[0].errorbar(self.xdata, self.residuals, self.yerr, 
                              fmt='bo', markersize=datsize)
            self.ax[0].axhline(0, color='k', linestyle='-')
            self.ax[0].plot(self.xp, [0, 0], 'bo', markersize=ptsize)
            self.ax[0].set_title('Residuals')
            
            # Set the limits
            self.ax[1].set_xlim(xl)
            self.ax[1].set_ylim(yl)
            
            # Calculate y-limits for residuals
            max_resid = max(np.max(np.abs(self.residuals)), 
                           np.max(self.yerr)) * 1.5
            if max_resid == 0:  # Handle case where all residuals are zero
                max_resid = 1.0
            self.ax[0].set_ylim(-max_resid, max_resid)
            
            # Set the figure's super title with padding
            if hasattr(self, 'name') and self.name:
                self.fig.suptitle(self.name, fontsize=12, y=0.98)
                
            # Adjust subplot spacing and update display
            self.fig.subplots_adjust(
                top=0.9,      # Leave space for title
                hspace=0.1,   # Space between subplots
                left=0.15,    # Left margin
                right=0.95,   # Right margin
                bottom=0.05   # Small bottom margin
            )
            self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for title and bottom
            self.fig.canvas.draw_idle()
            
            # Update output
            self.out.clear_output(wait=True)
            with self.out:
                self.print_output("", self.chi2)
                
            # Save state
            self.save_state()
            
        except Exception as e:
            print(f"Error updating display: {e}")
            if hasattr(self, 'out'):
                with self.out:
                    print(f"Detailed error: {str(e)}")
    
    def reset_parameters(self, button=None):
        """Reset parameters to initial values and clear cache"""
        try:
            # Remove the cache file if it exists
            if hasattr(self, 'filename') and os.path.exists(self.filename):
                os.remove(self.filename)
                
            # Reset parameters to initial values
            self.xp = np.array([np.min(self.xdata), np.max(self.xdata)])
            self.yp = np.array([np.average(self.ydata), np.average(self.ydata)])
            self.yoff = -4e9
            
            # Force update of slope and intercept based on new xp/yp
            self.slope = (self.yp[1] - self.yp[0]) / (self.xp[1] - self.xp[0]) if (self.xp[1] != self.xp[0]) else 0
            self.intercept = self.yp[0] - self.slope * self.xp[0]
            
            # Update the text boxes to reflect the new values
            if hasattr(self, 'slopewidget'):
                self.slopewidget.unobserve(self.slope_changed, names='value')
                self.slopewidget.value = par_form.format(self.slope)
                self.slopewidget.observe(self.slope_changed, names='value')
                
            if hasattr(self, 'intwidget'):
                self.intwidget.unobserve(self.int_changed, names='value')
                self.intwidget.value = par_form.format(self.intercept)
                self.intwidget.observe(self.int_changed, names='value')
                
            if hasattr(self, 'offwidget'):
                self.offwidget.unobserve(self.offset_changed, names='value')
                self.offwidget.value = par_form.format(self.yoff)
                self.offwidget.observe(self.offset_changed, names='value')
            
            # Recalculate the fit
            self.calc_fit()
            
            # Update the display
            self.update_displays(initial_scale=True)
            
            with self.out:
                print("Parameters have been reset to initial values.")
                
        except Exception as e:
            with self.out:
                print(f"Error resetting parameters: {e}")
    
    def __del__(self):
        """Clean up resources when the object is deleted"""
        try:
            if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
                plt.close(self.fig)
            if hasattr(self, 'out'):
                self.out.close()
        except:
            pass
    
    def calc_yp(self): # calculate the yp values
        self.yp = self.xp * self.slope + self.intercept
       
        
    def slope_changed(self, change):
        if self.no_recur:
            return
        #  check that its a valid number, if not reset:
        # with self.out:
        #     print("in slope_changed")
        try:
            new_val = float(change.new)
            self.slope = new_val
            self.no_recur = True
            self.slopewidget.value = par_form.format(self.slope)
            self.no_recur = False
        except:
            self.no_recur = True
            self.slopewidget.value = par_form.format(self.slope)
            self.no_recur = False
            return
        # set new yp values and update display
        self.calc_yp()
        self.calc_fit()
        self.update_displays()
        
    def int_changed(self, change):
        if self.no_recur:
            return
        try:
            new_val = float(change.new)
            self.intercept = new_val
            self.no_recur = True
            self.intwidget.value = par_form.format(self.intercept)
            self.no_recur = False
        except:
            self.no_recur = True
            self.intwidget.value = par_form.format(self.intercept)
            self.no_recur = False
            return
        self.calc_yp()
        self.calc_fit()
        self.update_displays()
        
    def offset_changed(self, change):
        if self.no_recur:
            return
        try:
            new_val = float(change.new)
            self.yoff = new_val
            self.no_recur = True
            self.offwidget.value = par_form.format(self.yoff)
            self.no_recur = False
        except:
            self.no_recur = True
            self.offwidget.value = par_form.format(self.yoff)
            self.no_recur = False
            return
        self.calc_yp()
        self.calc_fit()
        self.update_displays()
        
        
    def calc_fit(self):
        self.slope = (self.yp[1]-self.yp[0])/(self.xp[1]-self.xp[0])
        self.intercept = self.yp[1]-self.slope*self.xp[1]

        if self.use_background:
            self.residuals = self.ydata - np.log(np.exp(self.intercept)*np.exp(self.xdata*self.slope)+np.exp(self.yoff))
            self.cyp = np.log(np.exp(self.intercept)*np.exp(self.cxp*self.slope)+np.exp(self.yoff))
        else:
            self.residuals = self.ydata - self.intercept-self.xdata*self.slope
            self.cyp = self.intercept+self.cxp*self.slope
    
    def print_output(self, message, chi2):
        with self.out:
            if message != "":
                print(message)
            if self.use_background:
                if self.slope != 0:
                    print("R0: %.6g, Attenuation Coef: %.6g, Background: %.6g"%(np.exp(self.intercept), -self.slope, np.exp(self.yoff)))
            else:
                if self.input_boxes == False:
                    print("slope: %.6g, intercept: %.6g"%(self.slope, self.intercept))
            if chi2:
                chi2val = np.sum(self.residuals*self.residuals/self.yerr/self.yerr) 
                if self.use_background:
                    chi2val = chi2val/(len(self.residuals) - 3)
                else:
                    chi2val = chi2val/(len(self.residuals) - 2)
                print("Chi2: %.4g"%(chi2val))
        
    def __init__(self, name, xdata, ydata, yerr, use_background, chi2, input_boxes, render_to_html=None):
        global objs, names
        
        # Store the render_to_html setting or the export_toggle reference
        if render_to_html is None:
            try:
                from html_export_widget import export_toggle
                self.export_toggle = export_toggle
                self._render_to_html = None  # Will check export_toggle.value when needed
            except (ImportError, AttributeError):
                self.export_toggle = None
                self._render_to_html = False
        else:
            self.export_toggle = None
            self._render_to_html = bool(render_to_html)

        # Input validation with proper error messages
        if not isinstance(xdata, np.ndarray):
            raise ValueError("xdata must be a numpy array")
        if not isinstance(ydata, np.ndarray):
            raise ValueError("ydata must be a numpy array")
        if not isinstance(yerr, np.ndarray):
            raise ValueError("yerr must be a numpy array")
        if xdata.ndim != 1 or ydata.ndim != 1 or yerr.ndim != 1:
            raise ValueError("Data arrays must be one-dimensional")
        if len(xdata) < 2:
            raise ValueError("Must have at least 2 data points!")
        if len(ydata) != len(xdata) or len(yerr) != len(xdata):
            raise ValueError(f"xdata, ydata and yerr must all have the same number of elements: {len(xdata)}, {len(ydata)}, {len(yerr)}")
        if not name:
            raise ValueError("Name must not be empty!")
        
        # Close any existing figure with the same name
        plt.close(name)
        
        # Clean up any existing widget with the same name
        self.cleanup_existing(name)
        
        plt.ioff()
        
        # Look for our object in the current module's list of objects:        
        found_old = False
        for i in range(len(objs)):
            if name == names[i]:
                self = objs[i]
                found_old = True
                message = "Used fit values from previous invocation"
 
        # Even if we found an old instance, update with current data
        self.xdata = xdata
        self.ydata = ydata
        self.yerr = yerr
        self.chi2 = chi2
        self.use_background = use_background
        self.input_boxes = input_boxes
        
        if not found_old:
            # We're creating a new object
            self.filename = get_file_name(name)
            self.no_recur = False
            
            # Add to tracking lists
            objs.append(self)
            names.append(name)
            
            # Try to load previous state
            loaded = False
            message = "No previous fit parameters"
            
            try:
                if os.path.exists(self.filename):
                    with open(self.filename, 'rb') as ff:
                        loaded_data = pickle.load(ff)
                        if len(loaded_data) == 3:  # Check if we have all required data
                            self.xp, self.yp, self.yoff = loaded_data
                            if not isinstance(self.xp, np.ndarray):
                                self.xp = np.array(self.xp)
                                self.yp = np.array(self.yp)
                            loaded = True
                            message = "Loaded fit values from file"
            except Exception as e:
                print(f"Warning: Could not load previous state: {e}")
            
            if not loaded:
                # Initialize with default values
                self.xp = np.array([np.min(xdata), np.max(xdata)])
                self.yp = np.array([np.average(ydata), np.average(ydata)])
                self.yoff = -4e9
                
        
        # Set up the plot data
        if self.use_background:
            self.cxp = np.linspace(np.min(xdata), np.max(xdata), 100)
            self.cyp = np.zeros(100)  # cxp, cyp are points for fitted curve
        else:
            self.cxp = np.array((np.min(xdata), np.max(xdata)))
            self.cyp = np.zeros(2)

        # Calculate initial fit
        self.calc_fit()
        
        # Create the figure with a unique name
        self.fig = plt.figure(name, clear=True)
        
        # Configure figure
        self.fig.set_figheight(self.fig.get_figheight() * 1.3)
        self.fig.suptitle(name, fontsize=12, y=0.95)  # Keep the title
        self.fig.tight_layout()
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        
        # Create subplots
        self.ax = self.fig.subplots(2, sharex=True, height_ratios=[1, 2])

        self.ax[0].errorbar(xdata,self.residuals,yerr, fmt='bo', markersize=datsize)
        self.ax[0].plot((np.min(xdata),np.max(xdata)),(0,0),'k-')
        self.ax[0].plot(self.xp,(0,0), "bo", markersize=ptsize)
        self.ax[0].title.set_text('Residuals')
        self.fig.canvas.mpl_connect("button_press_event", self.onclick) # could do motion_notify_event, on_move(onmove?)
        self.line = Line2D(self.cxp,self.cyp)
        self.line2 = Line2D(self.xp,self.yp,marker='o',linestyle='',markersize=ptsize)
        self.ax[1].title.set_text(name)
        self.ax[1].errorbar(xdata,ydata,yerr, fmt='ro', label = "Data",markersize=datsize)
        self.ax[1].add_line(self.line)
        self.ax[1].add_line(self.line2)
        self.line.set_label("Fit")
        self.out = widgets.Output()
        
        self.button = widgets.RadioButtons(options=["Linear Slope/Intercept","Floor"]) # we look up its value later even if it doesn't appear.

        # We use text widgets so we can format numbers reasonably. with FloatText there's no way to control the formatting.
        self.slopewidget=widgets.Text(value=par_form.format(self.slope), description='Slope:', continuous_update=False)
        self.intwidget=widgets.Text(value=par_form.format(self.intercept), description='Intercept:', disabled=False, continuous_update=False)
        self.offwidget=widgets.Text(value=par_form.format(self.yoff), description='Floor:', disabled=False, continuous_update=False)
        #self.slopewidget=widgets.FloatText(value=self.slope, description='Slope:')
        #self.intwidget=widgets.FloatText(value=self.intercept, description='Intercept:')
        #self.offwidget=widgets.FloatText(value=self.yoff, description='Y offset:')

        self.slopewidget.observe(self.slope_changed, names='value')
        self.intwidget.observe(self.int_changed, names='value')
        self.offwidget.observe(self.offset_changed, names='value')
        
        # Add reset button
        self.reset_button = widgets.Button(description='Reset', 
                                         button_style='danger',
                                         tooltip='Reset parameters and clear cache')
        self.reset_button.on_click(self.reset_parameters)
        
        if self.use_background:
            if self.input_boxes:
                app = widgets.AppLayout(
                    header=self.fig.canvas,
                    left_sidebar=widgets.HBox([
                        self.slopewidget, 
                        self.intwidget, 
                        self.offwidget,
                        self.reset_button
                    ]),
                    footer=widgets.HBox([self.button, self.out]),
                    pane_heights=[12, 1, 1], grid_gap="1px", align_items='center',
                    pane_widths=[1, 0, 20])
            else:
                app = widgets.AppLayout(
                    header=self.fig.canvas,
                    left_sidebar=self.button,            
                    right_sidebar=self.out,
                    pane_heights=[12, 1, 0], grid_gap="1px", align_items='center',
                    pane_widths=[1, 0, 20])
        else: # just a line, no background
            if self.input_boxes:
                app = widgets.AppLayout(
                    header=self.fig.canvas,
                    footer=self.out,
                    left_sidebar=widgets.HBox([
                        self.slopewidget, 
                        self.intwidget,
                        self.reset_button
                    ]),
                    pane_heights=[12, 1, 1], grid_gap="1px", align_items='center',
                    pane_widths=[20, 0, 1])
            else:
                app = widgets.AppLayout(
                    header=self.fig.canvas,
                    left_sidebar=widgets.VBox([
                        self.out,
                        self.reset_button
                    ]),
                    pane_heights=[12, 1, 0], grid_gap="1px", align_items='center',
                    pane_widths=[20, 0, 1])
        
        # Check if we should render as HTML
        if self._render_to_html if self.export_toggle is None else self.export_toggle.value:
            from IPython.display import HTML, display as ipy_display
            import io
            import base64
            
            # Close the interactive figure to prevent it from showing up
            plt.close(self.fig)
            
            # Ensure we have proper scaling before rendering to HTML
            self.update_displays(initial_scale=True)
            plt.pause(0.1)  # Give matplotlib a moment to update the figure
            
            # Save the original titles
            orig_suptitle = self.fig._suptitle.get_text() if hasattr(self.fig, '_suptitle') and self.fig._suptitle is not None else ""
            
            # Set the main title (only if there isn't one already)
            if not orig_suptitle:
                self.fig.suptitle(name, y=0.98)
            
            # Ensure both subplots have proper scaling
            for i, ax in enumerate(self.fig.axes):
                if i == 1:  # Main plot
                    ax.set_title('Fit Plot')
                    
                    # Get y-values from data points (most important for scaling)
                    y_values = list(self.ydata)
                    
                    # Include error bars in the y-range calculation
                    yerr = self.yerr if hasattr(self, 'yerr') else np.zeros_like(self.ydata)
                    yerr = np.where(np.isnan(yerr), 0, yerr)  # Handle any NaN values
                    
                    # Calculate data range and add small padding
                    data_min = np.min(self.ydata - yerr)
                    data_max = np.max(self.ydata + yerr)
                    data_range = data_max - data_min
                    
                    # Add small padding (5% of range or 10% of max value, whichever is smaller)
                    padding = min(data_range * 0.05, np.max(np.abs([data_min, data_max])) * 0.1)
                    if padding == 0:  # In case all values are the same
                        padding = 0.1 * np.abs(data_max) if data_max != 0 else 0.1
                    
                    # Set the y-limits based on data range and padding
                    y_min = data_min - padding
                    y_max = data_max + padding
                    
                    # Ensure we don't go below zero if all data is positive
                    if np.min(y_values) >= 0:
                        y_min = max(0, y_min)
                    
                    # Apply the limits
                    ax.set_ylim(y_min, y_max)
                    
                    # No need to include fit or anchor points in y_values
                    # as we're setting the limits directly based on data
                    
                    # Get fit line data (excluding anchor points)
                    for line in ax.get_lines():
                        if line.get_label() == 'Fit':
                            y_fit = line.get_ydata()
                            if len(y_fit) > 0:
                                y_values.extend(y_fit)
                
                elif i == 0:  # Residuals plot
                    # For residuals, we want to center around zero
                    residuals = []
                    
                    # Get residuals from errorbar plot
                    for collection in ax.collections:
                        if hasattr(collection, '_y'):
                            residuals.extend(collection._y)
                            
                            # Include error bars if present
                            if hasattr(collection, '_yerr'):
                                yerr = collection._yerr
                                if yerr is not None:
                                    if isinstance(yerr, tuple):
                                        residuals.extend(collection._y - yerr[0])
                                        residuals.extend(collection._y + yerr[1])
                                    else:
                                        residuals.extend(collection._y - yerr)
                                        residuals.extend(collection._y + yerr)
                    
                    if residuals:
                        max_resid = max(np.max(np.abs(residuals)) * 1.2, 1e-10)
                        ax.set_ylim(-max_resid, max_resid)
                    continue
                
                # Skip setting y-limits here as we're setting them directly in the main plot section
                # This prevents the second scaling pass from overriding our carefully calculated limits
                if i != 1:  # Only process non-main plots here
                    if y_values:
                        y_min, y_max = np.min(y_values), np.max(y_values)
                        y_range = y_max - y_min
                        padding = max(y_range * 0.1, 0.1)  # Minimum 0.1 padding
                        ax.set_ylim(y_min - padding, y_max + padding)
                
            # Keep other subplot titles as they are
            
            try:
                # Set the scale factor (1.0 = default size, 0.5 = half size, 2.0 = double size)
                scale_factor = 0.8  # Adjust this value as needed
                base_width, base_height = 7.0, 8.0  # Base dimensions in inches
                new_width = base_width * scale_factor
                new_height = base_height * scale_factor
                
                # Set the new figure size
                self.fig.set_size_inches(new_width, new_height)
                
                # Adjust font sizes based on scale factor
                base_font_size = 14  # Base font size at scale 1.0
                font_scale = min(1.0, scale_factor * 1.2)  # Cap font scaling at 1.2x
                font_size = max(6, base_font_size * font_scale)  # Minimum font size of 6
                
                for ax in self.fig.axes:
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
                               ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(font_size)
                
                self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # Save the current figure to a buffer with higher DPI for better quality
                buf = io.BytesIO()
                self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=120,
                              facecolor=self.fig.get_facecolor())
                buf.seek(0)
                
                # Generate the output text
                import io
                import contextlib
                
                # Create a buffer to capture the standard output
                output_buffer = io.StringIO()
                with contextlib.redirect_stdout(output_buffer):
                    self.print_output("", self.chi2)
                output_text = output_buffer.getvalue().strip()
                
                # Get current parameter values
                if self.use_background:
                    params_text = f"""
                    Current Parameters:
                    - Slope: {self.slope:.6g}
                    - Intercept: {self.intercept:.6g}
                    - Floor: {self.yoff:.6g}
                    """
                else:
                    params_text = f"""
                    Current Parameters:
                    - Slope: {self.slope:.6g}
                    - Intercept: {self.intercept:.6g}
                    """
                
                # Combine the parameter values with the output text
                full_output = f"{params_text}\n\n{output_text}"
                
                # Create HTML with the image and text output, aligned left and properly scaled
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                container_width = new_width * 100  # Convert inches to pixels (approximate)
                
                html_content = f"""
                <div style="font-family: Arial, sans-serif; width: 100%; max-width: {container_width}px; 
                            margin: 0; text-align: left; font-size: {font_size}px;">
                    <img src='data:image/png;base64,{img_str}' 
                         style='width: 100%; height: auto; display: block;'/>
                    <div style='margin: 12px 0; padding: 12px; background: #f8f9fa; 
                              border-radius: 4px; font-family: monospace; white-space: pre; 
                              font-size: {font_size}px; border-left: 4px solid #007bff;'>
                        {full_output}
                    </div>
                </div>
                """
                
                # Display the HTML directly
                ipy_display(HTML(html_content))
                return None
                
            finally:
                # Restore the original suptitle if we modified it
                if not orig_suptitle and hasattr(self.fig, '_suptitle'):
                    self.fig._suptitle = None
        else:
            # Only show the interactive widget if we're not in HTML export mode
            if not (self._render_to_html if self.export_toggle is None else self.export_toggle.value):
                # Clear any existing output and display the widget
                clear_output(wait=True)
                display(app)
                
                # Print initial output and set the current axis
                self.print_output(message, self.chi2)
                plt.sca(self.ax[1])
                
                # Initial display with proper scaling
                self.update_displays(initial_scale=True)
                
                # Ensure the plot is drawn
                self.fig.canvas.draw_idle()
            return None

    def onclick(self, event):

        # print(os.environ['JPY_SESSION_NAME']) - this gets me my current file name.
        # save state in same path but .FILENAME-hashedPLOTNAME
            
        # button.index tells the state of the radio buttons.
        with self.out:
            # print(event.xdata,event.ydata)
            # print(event)
            if event.inaxes != self.ax[0] and event.inaxes != self.ax[1]:
                return
        
        if event.button != 1: # we only look at the left button.
            return
            
        # if you click outside the axes bad things happen Should never happen now?
        if not isinstance(event.xdata, float):
            return
        if not isinstance(event.ydata, float):
            return
           
        if self.button.index == 0 or self.use_background == False:  # doing a point on the line.
            # which point? Use the closer one. Need to scale distances by limits!
            if event.inaxes == self.ax[1]: # main plot            
                xl = self.ax[1].get_xlim()
                yl = self.ax[1].get_ylim()
                xs = abs(xl[1]-xl[0])
                ys = abs(yl[1]-yl[0])
                d0 = (event.xdata-self.xp[0])*(event.xdata-self.xp[0])/xs/xs +\
                            (event.ydata-self.yp[0])*(event.ydata-self.yp[0])/ys/ys
                d1 = (event.xdata-self.xp[1])*(event.xdata-self.xp[1])/xs/xs +\
                            (event.ydata-self.yp[1])*(event.ydata-self.yp[1])/ys/ys
                if (d0 < d1):
                    self.xp[0] = event.xdata
                    self.yp[0] = event.ydata
                else:
                    self.xp[1] = event.xdata
                    self.yp[1] = event.ydata
            elif event.inaxes == self.ax[0]: # in residuals
                if abs(event.xdata-self.xp[0]) < abs(event.xdata-self.xp[1]):
                    # modify xp0
                    self.xp[0] = event.xdata
                    self.yp[0] = event.ydata + self.slope * self.xp[0] + self.intercept
                else:
                    self.xp[1] = event.xdata
                    self.yp[1] = event.ydata + self.slope * self.xp[1] + self.intercept
        elif self.button.index == 1: # doing offset
            if event.inaxes == self.ax[1]: # main plot
                self.yoff = event.ydata
            elif event.inaxes == self.ax[0]: # clicked in residuals
                self.yoff = event.ydata + np.log(np.exp(self.intercept)*np.exp(event.xdata*self.slope)+np.exp(self.yoff))
        # common
        
        self.calc_fit()
        self.no_recur = True
        self.slopewidget.value=par_form.format(self.slope)
        self.intwidget.value=par_form.format(self.intercept)
        self.offwidget.value=par_form.format(self.yoff)
        # self.slopewidget.value=self.slope
        # self.intwidget.value=self.intercept
        # self.offwidget.value=self.yoff
        self.no_recur = False
        self.update_displays()
                
class line(generic_fit_with_background):
    """Class for creating fit objects for straight line fit.

    Parameters:
    name: a unique name that is used as a plot title as well as for tagging the fit parameters
    xdata: x values of the data
    ydata: y values
    yerr: uncertainties in y values
    use_background: a boolean. If false we just do a linear fit, if true the function is
            np.log(np.exp(self.intercept)*np.exp(self.cxp*self.slope)+np.exp(self.yoff))
    chi2: if True, show chi-squared value
    input_boxes: a boolean. If True display boxes that allow manual setting of parameters
    render_to_html: a boolean. If True, returns a static HTML representation instead of an interactive widget.
                   If None, will use the value of export_toggle.value if available.
    """
    def __init__(self, name, xdata, ydata, yerr, chi2=False, input_boxes=True, render_to_html=None):
        super().__init__(name, xdata, ydata, yerr, False, chi2, input_boxes, render_to_html)

class with_background(generic_fit_with_background):
    """Class for creating fit objects for radiation experiment with background

    Parameters:
    name: a unique name that is used as a plot title as well as for tagging the fit parameters
    xdata: x values of the data
    ydata: y values
    yerr: uncertainties in y values
    use_background: a boolean. If false we just do a linear fit, if true the function is
            np.log(np.exp(self.intercept)*np.exp(self.cxp*self.slope)+np.exp(self.yoff))
    chi2: if True, show chi-squared value
    input_boxes: a boolean. If True display boxes that allow manual setting of parameters
    render_to_html: a boolean. If True, returns a static HTML representation instead of an interactive widget.
                   If None, will use the value of export_toggle.value if available.
    """
    def __init__(self, name, xdata, ydata, yerr, chi2=False, input_boxes=True, render_to_html=None):
        super().__init__(name, xdata, ydata, yerr, True, chi2, input_boxes, render_to_html)
