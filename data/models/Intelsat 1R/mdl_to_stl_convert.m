% Open the Simulink Model
model_name = 'hs702';
open_system(model_name);

% Export Simscape Multibody model to an XML file
smexportonshape(model_name, 'exported_geometry.xml');

% Convert XML to STL using an external CAD tool or MATLAB (if possible)
fprintf('Exported geometry saved as exported_geometry.xml. Use CAD software to convert to STL.\n');
