# workbench GUI for geometryBuilder

workbench is a graphical user interface used to build models using geomertyBuilders features. workbench loads two windows on launch, the file editor and interface, and the renderer. These are used to build the model quickly and display how the model will look in satlight.

## Interface

The interface allows components to be added with the same settings as in geometryBuilder. 


### Save file

At the top of the interface are the inputs for the file path and name to be loaded or saved to. The load file button can be used to display a model if the file has already been created, but adding any components will overwrite this.

### Adding components

There are 4 main sections containing inputs to add components. All inputs have default values if left blank.

- Universal settings: This section includes scaling, translations, rotations, and materials that apply to a component. Only materials included in materialLibrary.txt are available for selection. Materials can be added by adding them to the file as they would be in an mtl file
- Subassemblies: File path and file name inputs to add premade files. File extension (.obj) should not be included in the file name. The x scale acts as the model scale factor. Materials do not influence imported models, this is taken from their mtl files. Click add subassembly button to add part
- Prisms: All settings for prisms can be input. Click add prims button to add part
- Dishes: All setting for parabolas can be input. No scale factors are used for parabolas. Click add dish button to add part

### Other UI elements

- Select a component: Once a part has been created, it will be available to select. Doing so will load its setting. If this does not happen, the load component button at the bottom of the interface will load the selected component
- Clear inputs: This button resets the part setting inputs 
- Save edits to component: A selected part will be updated with any changes to its settings. The part type cannot be changed after creation e.g. a prism cannot be changed to a dish
- Delete selected component: Marks the selected part as deleted and stops it being generated in the geometry
- Generate geometry: Regenerates the geometry based on the created parts and reloads the model into the renderer
- Close: Closes the interface and rendering window

## Renderer

The renderer displays the loaded file. Axes are displayed in the top right with +X (right) in red, +Y (forward) in green, and +Z (up) in blue.

### Controls

- Rotation: The model can be rotated by moving the mouse with middle click held down
- Zoom: Scrolling controls the zoom level 
- Moving: The view of the model can be moved by moving the mouse with left click held down

In the top right of the interface is the reset view button. This returns the view to front on at the original position and zoom level

### Render options

In the bottom left of the interface, there are options for whether the renderer displays vertices, edges, or faces. It is recommended to use faces for the best user experience and the most optimised rendering.
