from math import cos, inf, pi
import tkinter as tk
from satlight import geometryBuilder

theta_x = 0
theta_y = 0
theta_z = 0
vertex_list = []
normal_list = []
normal_index = []
faces = []
x_range = [0, 0]
y_range = [0, 0]
z_range = [0, 0]
window_size = 750
current_pos = [0, 0]
mousexz = [0, 0]
priordeltas = [0, 0]
deltax = 0
deltaz = 0
zoom = 1
shift = [0, 0]
shift_clicked_pos = [0, 0]
total_shift = [0, 0]
shift_delta_x = 0
shift_delta_z = 0

component_type = [] #1 subassembly     2 prism     3 parabola       0 errors        code*10 for deleted
subassembly_path = []
subassembly_name = []
component_settings = []
type_settings = []

root = tk.Tk()
root.title("model render")
root.geometry(str(window_size) + "x" + str(window_size) + "+500+0")
canvas = tk.Canvas(root, height = window_size, width = window_size, bg="black")

canvas.pack()

UI = tk.Tk()
UI.title("UI")
UI.geometry("500x" + str(window_size) + "+0+0")

def parallax_correction(vertex):

    x = vertex[0]
    y = vertex[1]
    z = vertex[2]

    try:
        scale = 1/y
    except:
        scale = inf
    x_corrected = x * scale
    z_corrected = z * scale

    display_vector = [x_corrected, z_corrected]
    return display_vector


def load_model(file):

    global x_range 
    global y_range
    global z_range
    global vertex_list
    global normal_list
    global normal_index
    global faces

    x_range = [0, 0]
    y_range = [0, 0]
    z_range = [0, 0]

    with open(file) as f:

        vertex_list = []
        face_list = []
        normal_list = []
        normal_index = []

        
        lines = f.readlines()

        for line in lines:

            if line.find('v ') != -1:

                space_count = 0
                digit = ""
                coords = []

                for char in line:

                    if char == ' ' or char == "\n":

                        space_count += 1
                        try:
                            coords.append(float(digit))
                            digit = ""
                        except:
                            pass
                    elif space_count > 0:
                        try:
                            digit += str(int(char))
                        except:
                            if char == '.':
                                digit += char
                            elif char == '-':
                                digit += char
                            elif char == 'e':
                                digit += char

                x = float(coords[0])
                y = float(coords[1])
                z = float(coords[2])

                if x > x_range[0]:
                    x_range[0] = x
                elif x < x_range[1]:
                    x_range[1] = x

                if y > y_range[0]:
                    y_range[0] = y
                elif y < y_range[1]:
                    y_range[1] = y

                if z > z_range[0]:
                    z_range[0] = z
                elif z < z_range[1]:
                    z_range[1] = z

                vertex = (x, y, z)

                vertex_list.extend([vertex])


            elif line.find('vn ') != -1:

                normal = []
                digit = ''
                space_count = 0

                for char in line:
                    if char == ' ' or char == '\n':
                        space_count += 1
                        try:
                            normal.append(float(digit))
                            digit = ""
                        except:
                            pass

                    elif space_count > 0:
                        try:
                            digit += str(int(char))
                        except:
                            if char == '.':
                                digit += char
                            elif char == '-':
                                digit += char
                            elif char == 'e':
                                digit += char

                normal_list.append(normal)



            elif line.find("f ") != -1:

                
                space_count = 0
                index_location = 0
                face = ""
                normal_pointer = ""
                corners = []

                for char in line:

                    if char == " " or char == '\n':

                        try:
                            corners.append(float(face))
                            face = ""
                        except:
                            pass
                        index_location = 0
                        space_count += 1

                    elif char == "/":

                        index_location += 1

                    elif index_location == 0 and space_count != 0:
                        face += char

                    elif index_location == 2 and space_count == 1:
                        normal_pointer += char


                face_list.extend([corners])
                normal_index.append(int(normal_pointer))


    return vertex_list, face_list

def dot_product(vec1, vec2):

    dot = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    return dot



def hex_generator(illumination):

    value = illumination * 255
    if value > 128:
        value = round(value - ((value - 128) / 3))
    elif value < 128:
        value = round(value + ((128 - value) / 4))
    digit = str(hex(value))[2:]
    if len(digit) == 1:
        digit = "0" + digit

    hexcode = "#" + digit + digit + digit
    return hexcode

def draw_points():

    global theta_x
    global theta_y
    global theta_z
    global vertex_list
    global normal_index
    global normal_list
    global faces
    global x_range
    global y_range
    global z_range
    global current_pos
    global zoom
    global shift

    scale = 1000 * zoom
    distance = 2

    drawn_points = []
    mean_depth = []
    mean_depth_index = []
    corrected_vertex = []
    canvas.delete('all')


    x_size = (abs(x_range[0]) + abs(x_range[1])) / 2
    y_size = (abs(y_range[0]) + abs(y_range[1])) / 2
    z_size = (abs(z_range[0]) + abs(z_range[1])) / 2

    x_centre = x_range[0] - x_size
    y_centre = y_range[0] - y_size
    z_centre = z_range[0] - z_size

    ordered_scales = [0, 0, 0]

    if x_size >= y_size and x_size >= z_size:

        ordered_scales[2] = x_size

        if y_size >= z_size:

            ordered_scales[1] = y_size
            ordered_scales[0] = z_size
        else:
            ordered_scales[1] = z_size
            ordered_scales[0] = y_size

    elif y_size >= x_size and y_size >= z_size:

        ordered_scales[2] = y_size

        if x_size >= z_size:

            ordered_scales[1] = x_size
            ordered_scales[0] = z_size
        else:
            ordered_scales[1] = z_size
            ordered_scales[0] = x_size

    elif z_size >= y_size and z_size >= x_size:

        ordered_scales[2] = z_size

        if y_size >= x_size:

            ordered_scales[1] = y_size
            ordered_scales[0] = x_size
        else:
            ordered_scales[1] = x_size
            ordered_scales[0] = y_size

    render_type = render_config.get()

    for i in range(len(vertex_list)):

        vertex = vertex_list[i]      

        if theta_x != 0:
            vertex = geometryBuilder.x_rotation(theta_x,vertex)
        if theta_y != 0:
            vertex = geometryBuilder.y_rotation(theta_y, vertex)
        if theta_z != 0:
            vertex = geometryBuilder.z_rotation(theta_z,vertex)

        x = (vertex[0] + x_centre) * scale + shift[0]
        z = (vertex[2] + z_centre) * scale + shift[1]
        
        y = vertex[1] + distance / 2 + distance * 2 * ordered_scales[2]

        vertex = [x,y,z]
        corrected_vertex.append(vertex)

        display_vertex = parallax_correction(vertex)

        display_vertex[0] += window_size / 2
        display_vertex[1] += window_size / 2
        drawn_points.append(display_vertex)
        #render vertecies
        if render_type == 1:
            canvas.create_line(display_vertex[0], display_vertex[1], display_vertex[0] + 1, display_vertex[1] + 1, fill = "red", width=2)
        
    #render edges
    if render_type == 2:
        for j in range(len(faces)):

            lines = faces[j]

            for k in range(len(lines) - 1):
                pos1 = int(lines[k]) - 1
                pos2 = int(lines[k + 1]) - 1
                coord1 = drawn_points[pos1]
                coord2 = drawn_points[pos2]
                canvas.create_line(coord1[0], coord1[1], coord2[0], coord2[1], fill = "blue", width = 2)

                if k == (len(lines) - 2):
                    pos0 = int(lines[0]) - 1
                    coord1 = drawn_points[pos0]
                    canvas.create_line(coord1[0], coord1[1], coord2[0], coord2[1], fill = "blue", width = 2)

    #render faces
    if render_type == 3:
        #find depth
        for k in range(len(faces)):

            lines = faces[k]

            if len(lines) == 3:
                coords1 = corrected_vertex[int(lines[0])-1]
                coords2 = corrected_vertex[int(lines[1])-1]
                coords3 = corrected_vertex[int(lines[2])-1]
                mean_depth.append((coords1[1] + coords2[1] + coords3[1]) / 3)
                mean_depth_index.append(k)

            if len(lines) == 4:
                coords1 = corrected_vertex[int(lines[0])-1]
                coords2 = corrected_vertex[int(lines[1])-1]
                coords3 = corrected_vertex[int(lines[2])-1]
                coords4 = corrected_vertex[int(lines[3])-1]
                mean_depth.append((coords1[1] + coords2[1] + coords3[1] + coords4[1]) / 4)
                mean_depth_index.append(k)

        #order depths
        try:
            mean_depth, mean_depth_index = zip(*sorted(zip(mean_depth, mean_depth_index)))
        except:
            pass

        #draws faces
        for j in range(len(faces)):
        
            furthest = len(mean_depth_index)
            face_index = mean_depth_index[furthest - j - 1]
            lines = faces[face_index]
            norm = normal_index[face_index]
            normal = normal_list[norm - 1]

            if theta_x != 0:
                normal = geometryBuilder.x_rotation(theta_x, normal)
            if theta_y != 0:
                normal = geometryBuilder.y_rotation(theta_y, normal)
            if theta_z != 0:
                normal = geometryBuilder.z_rotation(theta_z, normal)

            direction = dot_product(normal, [0, -1, 0])

            if direction > 0.001:
                shade = hex_generator(direction)

                if len(lines) == 3:
                    coords1 = drawn_points[int(lines[0])-1]
                    coords2 = drawn_points[int(lines[1])-1]
                    coords3 = drawn_points[int(lines[2])-1]
                    canvas.create_polygon(coords1[0],coords1[1], coords2[0],coords2[1], coords3[0],coords3[1],fill=shade)

                if len(lines) == 4:
                    coords1 = drawn_points[int(lines[0])-1]
                    coords2 = drawn_points[int(lines[1])-1]
                    coords3 = drawn_points[int(lines[2])-1]
                    coords4 = drawn_points[int(lines[3])-1]
                    canvas.create_polygon(coords1[0],coords1[1], coords2[0],coords2[1], coords3[0],coords3[1], coords4[0],coords4[1],fill=shade)



def next_frame():

    draw_points()
    canvas.after(50, next_frame)


def model_rotation(x_angle = 0, y_angle = 0, z_angle = 0):

    global vertex_list

    for i in range(len(vertex_list)):

        vertex = vertex_list[i]

        if x_angle != 0:
            vertex = geometryBuilder.x_rotation(x_angle,vertex)
        if y_angle != 0: 
            vertex = geometryBuilder.y_rotation(y_angle,vertex)
        if z_angle != 0: 
            vertex = geometryBuilder.z_rotation(z_angle,vertex)

        vertex_list[i] = vertex

#mouse rotation
def rotations(event):

    global current_pos
    global theta_x
    global theta_y
    global theta_z
    global mousexz
    global priordeltas
    global deltax
    global deltaz

    current_pos = [event.x, event.y]

    normalised_theta_z = theta_z
    while normalised_theta_z > 360:
        normalised_theta_z -= 360
    while normalised_theta_z < 0:
        normalised_theta_z += 360
    normalised_theta_z /= 90

    deltaz = current_pos[0] + mousexz[0] + priordeltas[0]

    if normalised_theta_z >= 1 and normalised_theta_z < 3:
        deltay = current_pos[1] - mousexz[1] + priordeltas[1]
    else:
        deltay = - current_pos[1] + mousexz[1] + priordeltas[1]

    if normalised_theta_z >= 0 and normalised_theta_z < 2:
        deltax = current_pos[1] - mousexz[1] + priordeltas[1]
    elif normalised_theta_z >= 2 and normalised_theta_z < 4:
        deltax = - current_pos[1] + mousexz[1] + priordeltas[1]

    side_ratio = abs(cos(2* pi* theta_z / 360))

    theta_x = deltay * side_ratio / 3
    theta_y = deltax * (1 - side_ratio) / 3
    theta_z = deltaz / 3
       

def button_up(event):
    global priordeltas
    global deltax
    global deltaz

    priordeltas = [deltaz, deltax]

def button_down(event):
    global mousexz
    mousexz[0] = event.x
    mousexz[1] = event.y

def zoomin(event):
    global zoom
    zoom += event.delta / (120 * 8)
    if zoom < 0:
        zoom = 0

def shift_model(event):
    global shift
    global shift_clicked_pos
    global total_shift
    global shift_delta_x
    global shift_delta_z

    shift_pos = [event.x, event.y]
    shift_delta_x = (shift_pos[0] - shift_clicked_pos[0]) * 2.7
    shift_delta_z = (shift_pos[1] - shift_clicked_pos[1]) * 2.7
    shift = [shift_delta_x + total_shift[0], shift_delta_z + total_shift[1]]


def shift_clicked(event):
    global shift_clicked_pos

    shift_clicked_pos = [event.x, event.y]

def shift_released(event):
    global total_shift
    global shift
    global shift_delta_x
    global shift_delta_z

    total_shift[0] += shift_delta_x
    total_shift[1] += shift_delta_z

canvas.bind("<B2-Motion>", rotations)
canvas.bind("<ButtonRelease-2>", button_up)
canvas.bind("<ButtonPress-2>", button_down)
canvas.bind("<MouseWheel>", zoomin)
canvas.bind("<B1-Motion>", shift_model)
canvas.bind("<ButtonPress-1>", shift_clicked)
canvas.bind("<ButtonRelease-1>", shift_released)


#UI functions
def close_app():
    root.destroy()
    UI.destroy()

def update_model():
    global vertex_list
    global faces

    input_filepath = path_in.get()
    input_name = name_in.get()
    try:
        vertex_list, faces = load_model(input_filepath + "\\" + input_name + ".obj")
        model_rotation(x_angle = 0, y_angle = 0)
       
    except:
        popup = tk.Toplevel()
        popup.title("Error")
        popup.geometry("250x100+500+300")
        tk.Label(popup, text="Error: No file with this address").place(x=40,y=35)

def home():
    global theta_x
    global theta_y
    global theta_z
    global priordeltas
    global zoom
    global shift
    global total_shift

    priordeltas = [0, 0]
    theta_x = 0
    theta_y = 0
    theta_z = 0
    zoom = 1
    shift = [0, 0]
    total_shift = [0, 0]

def create_geometry():
    
    global component_list
    global component_type
    global type_settings
    global component_settings

    input_filepath = path_in.get()
    input_name = name_in.get()
    try:
        geometryBuilder.setup(input_filepath + "\\", input_name)
    except:
        popup = tk.Toplevel()
        popup.title("Error")
        popup.geometry("250x100+500+300")
        tk.Label(popup, text="Error: Enter a valid filepath ").place(x=40,y=35)

    for i in range(len(component_list)):

        setting = component_settings[i]
        typesetting = type_settings[i]

        if component_type[i] == 1:
            
            try:
                geometryBuilder.import_model(typesetting[0], typesetting[1] + "\\", scale = setting[0], x_offset = setting[3], y_offset = setting[4], z_offset = setting[5] , x_angle = setting[6], y_angle = setting[7], z_angle = setting[8])
            except:
                popup = tk.Toplevel()
                popup.title("Error")
                popup.geometry("250x100+500+300")
                try:
                    tk.Label(popup, text="Error: Missing component " + str(typesetting[0])).place(x=40,y=35)
                except:
                    tk.Label(popup, text="Error: Missing component").place(x=40,y=35)


        elif component_type[i] == 2:

            try:
                geometryBuilder.prism_geometry(name = typesetting[0], side_count = typesetting[1], radius = typesetting[2], height = typesetting[3], taper = typesetting[4], is_hollow = typesetting[5], wall_thickness = typesetting[6], x_scale = setting[0], y_scale = setting[1], z_scale = setting[2], x_offset = setting[3], y_offset = setting[4], z_offset = setting[5] , x_angle = setting[6], y_angle = setting[7], z_angle = setting[8])
            except:
                popup = tk.Toplevel()
                popup.title("Error")
                popup.geometry("250x100+500+300")
                tk.Label(popup, text="Error: No save file " + str(typesetting[0])).place(x=40,y=35)


        elif component_type[i] == 3:

            try:
                geometryBuilder.parabola_geometry(name = typesetting[0], radius = typesetting[1], height = typesetting[2], segments = typesetting[3], fidelity = typesetting[4], x_offset = setting[3], y_offset = setting[4], z_offset = setting[5] , x_angle = setting[6], y_angle = setting[7], z_angle = setting[8])
                geometryBuilder.parabola_geometry(name = '', radius = typesetting[1], height = -typesetting[2], segments = typesetting[3], fidelity = typesetting[4], x_offset = setting[3], y_offset = setting[4], z_offset = setting[5] - 0.01 , x_angle = setting[6] + 180, y_angle = setting[7], z_angle = setting[8])
                geometryBuilder.prism_geometry(name = '',side_count = typesetting[4], radius = typesetting[1], height = 0.01, taper = 1, is_hollow = True, wall_thickness = 0.001, x_offset = setting[3], y_offset = setting[4], z_offset = setting[5] + typesetting[2] - 0.0005, x_angle = setting[6], y_angle = setting[7], z_angle = setting[8] + (360 / typesetting[4]) / 2)
            except:
                popup = tk.Toplevel()
                popup.title("Error")
                popup.geometry("250x100+500+300")
                tk.Label(popup, text="Error: No save file " + str(typesetting[0])).place(x=40,y=35)


    update_model()

def update_list(name):
        
    global component_list
    global component_number
    global component_menu

    current_length = len(component_list)
    component_list.append((str(current_length + 1) + "    " + name))
    component_menu.destroy()
    component_menu = tk.OptionMenu(UI, component_number, *component_list)
    component_menu.place(x = 5, y = 120)



def save_settings():

    global component_settings


    try:
        xscale = float(x_scale_input.get())
    except:
        xscale = 1
    try:
        yscale = float(y_scale_input.get())
    except:
        yscale = 1
    try:
        zscale = float(z_scale_input.get())
    except:
        zscale = 1


    try:
        xoffset = float(x_offset_input.get())
    except:
        xoffset = 0
    try:
        yoffset = float(y_offset_input.get())
    except:
        yoffset = 0
    try:
        zoffset = float(z_offset_input.get())
    except:
        zoffset = 0


    try:
        xrotation = float(x_rotation_input.get())
    except:
        xrotation = 0
    try:
        yrotation = float(y_rotation_input.get())
    except:
        yrotation = 0
    try:
        zrotation = float(z_rotation_input.get())
    except:
        zrotation = 0

    setting = (xscale, yscale, zscale, xoffset, yoffset, zoffset, xrotation, yrotation, zrotation)
    component_settings.append(setting)


def add_subassembly():

    global component_type
    global type_settings

    name = str(add_name.get())
    update_list(name)
    save_settings()

    component_type.append(1)

    try:
        subname = str(add_name.get())
    except:
        subname = "<insert name>"
    try:
        subfile = str(add_file.get())
    except:
        subfile = "<no file>"

    setting = (subname, subfile)
    type_settings.append(setting)
   

def load_component():

    global component_number
    global component_type
    global component_settings
    global ishollow

    selected = component_number.get()
    item_id = ""
    
    for char in selected:
        try:
            item_id += str(int(char))
        except:
            break

    try:
        id_num = int(item_id) - 1
        type_id = component_type[id_num]
    except:
        type_id = 0

    setting = component_settings[id_num]

    if type_id > 9:
        type_id /= 10

    x_scale_input.delete(0, tk.END)
    x_scale_input.insert(0, setting[0])
    y_scale_input.delete(0, tk.END)
    y_scale_input.insert(0, setting[1])
    z_scale_input.delete(0, tk.END)
    z_scale_input.insert(0, setting[2])

    x_offset_input.delete(0, tk.END)
    x_offset_input.insert(0, setting[3])
    y_offset_input.delete(0, tk.END)
    y_offset_input.insert(0, setting[4])
    z_offset_input.delete(0, tk.END)
    z_offset_input.insert(0, setting[5])

    x_rotation_input.delete(0, tk.END)
    x_rotation_input.insert(0, setting[6])
    y_rotation_input.delete(0, tk.END)
    y_rotation_input.insert(0, setting[7])
    z_rotation_input.delete(0, tk.END)
    z_rotation_input.insert(0, setting[8])


    setting = type_settings[id_num]

    if type_id == 1:

        add_file.delete(0, tk.END)
        add_file.insert(0, setting[1])
        add_name.delete(0, tk.END)
        add_name.insert(0, setting[0])

        
        #clear other components
        prism_name_entry.delete(0, tk.END)
        side_count_entry.delete(0, tk.END)
        prism_radius_entry.delete(0, tk.END)
        prism_height_entry.delete(0, tk.END)
        taper_entry.delete(0, tk.END)
        hollow_check.deselect()
        wall_thickness_entry.delete(0, tk.END)
        dish_name_entry.delete(0, tk.END)
        dish_radius_entry.delete(0, tk.END)
        dish_height_entry.delete(0, tk.END)
        dish_segments_entry.delete(0, tk.END)
        dish_fidelity_entry.delete(0, tk.END)

    if type_id == 2:

        prism_name_entry.delete(0, tk.END)
        prism_name_entry.insert(0, setting[0])
        side_count_entry.delete(0, tk.END)
        side_count_entry.insert(0, setting[1])
        prism_radius_entry.delete(0, tk.END)
        prism_radius_entry.insert(0, setting[2])
        prism_height_entry.delete(0, tk.END)
        prism_height_entry.insert(0, setting[3])
        taper_entry.delete(0, tk.END)
        taper_entry.insert(0, setting[4])
        hollow = setting[5]
        if hollow == True:
            hollow_check.select()
        else:
            hollow_check.deselect()
        wall_thickness_entry.delete(0, tk.END)
        wall_thickness_entry.insert(0, setting[6])


        #clear other components
        add_file.delete(0, tk.END)
        add_name.delete(0, tk.END)
        dish_name_entry.delete(0, tk.END)
        dish_radius_entry.delete(0, tk.END)
        dish_height_entry.delete(0, tk.END)
        dish_segments_entry.delete(0, tk.END)
        dish_fidelity_entry.delete(0, tk.END)


    if type_id == 3:

        dish_name_entry.delete(0, tk.END)
        dish_name_entry.insert(0, setting[0])
        dish_radius_entry.delete(0, tk.END)
        dish_radius_entry.insert(0, setting[1])
        dish_height_entry.delete(0, tk.END)
        dish_height_entry.insert(0, setting[2])
        dish_segments_entry.delete(0, tk.END)
        dish_segments_entry.insert(0, setting[3])
        dish_fidelity_entry.delete(0, tk.END)
        dish_fidelity_entry.insert(0, setting[4])


        #clear other components
        add_file.delete(0, tk.END)
        add_name.delete(0, tk.END)
        prism_name_entry.delete(0, tk.END)
        side_count_entry.delete(0, tk.END)
        prism_radius_entry.delete(0, tk.END)
        prism_height_entry.delete(0, tk.END)
        taper_entry.delete(0, tk.END)
        hollow_check.deselect()
        wall_thickness_entry.delete(0, tk.END)


def add_prism():
    
    global component_type
    global type_settings
    global ishollow

    try:
        name = str(prism_name_entry.get())
    except:
        name = "<insert name>"

    update_list(name)
    save_settings()

    component_type.append(2)

    try:
        sidecount = int(side_count_entry.get())
    except:
        sidecount = 4
    try:
        radius = float(prism_radius_entry.get())
    except:
        radius = 0.7071
    try:
        height = float(prism_height_entry.get())
    except:
        height = 1

    try:
        taper = float(taper_entry.get())
    except:
        taper = 1

    hollow = check_hollow()
    try:
        wallthickness = float(wall_thickness_entry.get())
    except:
        wallthickness = 0.1


    setting = (name, sidecount, radius, height, taper, hollow, wallthickness)
    type_settings.append(setting)


def delete_component():
    
    global component_type
    global component_list
    global component_menu
    global component_number

    selected = component_number.get()
    item_id = ""
    
    for char in selected:
        try:
            item_id += str(int(char))
        except:
            break

    try:
        id_num = int(item_id) - 1
        component_type[id_num] *= 10

        component_list[id_num] = str(id_num + 1) + "    [DELETED]"

        component_menu.destroy()
        component_menu = tk.OptionMenu(UI, component_number, *component_list)
        component_menu.place(x = 5, y = 120)
    except:
        pass


def add_dish():

    global component_type
    global type_settings

    try:
        name = str(dish_name_entry.get())
    except:
        name = "<insert name>"

    update_list(name)
    save_settings()

    component_type.append(3)

    try:
        radius = float(dish_radius_entry.get())
    except:
        radius = 1

    try:
        height = float(dish_height_entry.get())
    except:
        height = 1

    try:
        segments = int(dish_segments_entry.get())
    except:
        segments = 15

    try:
        fidelity = int(dish_fidelity_entry.get())
    except:
        fidelity = 36


    setting = (name, radius, height, segments, fidelity)
    type_settings.append(setting)



def update_component():
    
    global component_number
    global component_type
    global component_settings
    global type_settings
    global component_list
    global component_menu

    selected = component_number.get()
    item_id = ""
    
    for char in selected:
        try:
            item_id += str(int(char))
        except:
            break

    try:
        id_num = int(item_id) - 1
        type_id = component_type[id_num]
    except:
        type_id = 0

    if type_id > 9:
        type_id /= 10
        component_type[id_num] = type_id




    try:
        xscale = float(x_scale_input.get())
    except:
        xscale = 1
    try:
        yscale = float(y_scale_input.get())
    except:
        yscale = 1
    try:
        zscale = float(z_scale_input.get())
    except:
        zscale = 1


    try:
        xoffset = float(x_offset_input.get())
    except:
        xoffset = 0
    try:
        yoffset = float(y_offset_input.get())
    except:
        yoffset = 0
    try:
        zoffset = float(z_offset_input.get())
    except:
        zoffset = 0


    try:
        xrotation = float(x_rotation_input.get())
    except:
        xrotation = 0
    try:
        yrotation = float(y_rotation_input.get())
    except:
        yrotation = 0
    try:
        zrotation = float(z_rotation_input.get())
    except:
        zrotation = 0

    setting = (xscale, yscale, zscale, xoffset, yoffset, zoffset, xrotation, yrotation, zrotation)
    component_settings[id_num] = setting

    if type_id == 1:

        try:
            subname = str(add_name.get())
        except:
            subname = "<insert name>"
        try:
            subfile = str(add_file.get())
        except:
            subfile = "<no file>"

        setting = (subname, subfile)
        type_settings[id_num] = setting

    if type_id == 2:


        try:
            name = str(prism_name_entry.get())
        except:
            name = "<insert name>"
        try:
            sidecount = int(side_count_entry.get())
        except:
            sidecount = 4
        try:
            radius = float(prism_radius_entry.get())
        except:
            radius = 0.7071
        try:
            height = float(prism_height_entry.get())
        except:
            height = 1
        try:
            taper = float(taper_entry.get())
        except:
            taper = 1
        hollow = check_hollow()
        try:
            wallthickness = float(wall_thickness_entry.get())
        except:
            wallthickness = 0.1


        setting = (name, sidecount, radius, height, taper, hollow, wallthickness)
        type_settings[id_num] = setting

    if type_id == 3:

        try:
            name = str(dish_name_entry.get())
        except:
            name = "<insert name>"
        try:
            radius = float(dish_radius_entry.get())
        except:
            radius = 1
        try:
            height = float(dish_height_entry.get())
        except:
            height = 1
        try:
            segments = int(dish_segments_entry.get())
        except:
            segments = 15
        try:
            fidelity = int(dish_fidelity_entry.get())
        except:
            fidelity = 36

        setting = (name, radius, height, segments, fidelity)
        type_settings[id_num] = setting


    component_list[id_num] = str(id_num + 1) + "    " + setting[0]

    component_menu.destroy()
    component_menu = tk.OptionMenu(UI, component_number, *component_list)
    component_menu.place(x = 5, y = 120)
    

def check_hollow():
    global ishollow
    hollow = ishollow.get()
    return hollow

#UI setup

#close
tk.Button(UI, text="Close", width = 15, command = close_app).place(x = 200, y = window_size - 50)

#update
tk.Button(UI, text="Load file", width = 16, command = update_model).place(x = 175, y = 20)

#home
tk.Button(UI, text = "Reset view", width = 15, command = home).place(x = 340, y = 20)

#info
tk.Label(UI, text = "Save and load file",  font = ("Ariel", 10)).place(x = 5, y = 35)
tk.Label(UI, text = "All measurements in metres, all angles in degrees", font = ("Ariel", 9)).place(x = 5, y = 165)

#file entry
tk.Label(UI, text = "File path:").place(x = 5, y = 60)
path_in = tk.Entry(UI, width = 50)
path_in.place(x = 75, y = 62)

tk.Label(UI, text = "File name:").place(x = 5, y = 90)
name_in = tk.Entry(UI, width = 20)
name_in.place(x = 75, y = 92)

#generate
tk.Button(UI, text = "Generate geometry", width = 20, command = create_geometry).place(x = 340, y = window_size - 51)

#component list
component_number = tk.StringVar(UI)
component_number.set("Select a component")
component_list = []
component_menu = tk.OptionMenu(UI, component_number, component_list)
component_menu.place(x = 5, y = 120)

tk.Button(UI, text = "Load selected component", width = 20, command = load_component).place(x = 170, y = 120)
tk.Button(UI, text = "Save edits to component", width = 21, command = update_component).place(x = 330, y = 120)
tk.Button(UI, text = "Delete selected component", width = 21, command = delete_component).place(x = 330, y = 150)

#   scales translations and rotations
#scales
tk.Label(UI, text = "x scale:").place(x = 5, y = 190)
tk.Label(UI, text = "y scale:").place(x = 5, y = 220)
tk.Label(UI, text = "z scale:").place(x = 5, y = 250)

x_scale_input = tk.Entry(UI, width = 15)
x_scale_input.place(x = 55, y = 192)
y_scale_input = tk.Entry(UI, width = 15)
y_scale_input.place(x = 55, y = 222)
z_scale_input = tk.Entry(UI, width = 15)
z_scale_input.place(x = 55, y = 252)

#translations
tk.Label(UI, text = "x offset:").place(x = 165, y = 190)
tk.Label(UI, text = "y offset:").place(x = 165, y = 220)
tk.Label(UI, text = "z offset:").place(x = 165, y = 250)

x_offset_input = tk.Entry(UI, width = 15)
x_offset_input.place(x = 220, y = 192)
y_offset_input = tk.Entry(UI, width = 15)
y_offset_input.place(x = 220, y = 222)
z_offset_input = tk.Entry(UI, width = 15)
z_offset_input.place(x = 220, y = 252)

#rotations
tk.Label(UI, text = "x rotation:").place(x = 325, y = 190)
tk.Label(UI, text = "y rotation:").place(x = 325, y = 220)
tk.Label(UI, text = "z rotation:").place(x = 325, y = 250)

x_rotation_input = tk.Entry(UI, width = 15)
x_rotation_input.place(x = 390, y = 192)
y_rotation_input = tk.Entry(UI, width = 15)
y_rotation_input.place(x = 390, y = 222)
z_rotation_input = tk.Entry(UI, width = 15)
z_rotation_input.place(x = 390, y = 252)



#add subassembly
tk.Label(UI, text = "Add subassembly", font = ("Ariel", 12)).place(x = 5, y = 300)

tk.Label(UI, text = "File path:").place(x = 5, y = 330)
add_file = tk.Entry(UI, width = 50)
add_file.place(x = 75, y = 332)

tk.Label(UI, text = "File name:").place(x = 5, y = 360)
add_name = tk.Entry(UI, width = 20)
add_name.place(x = 75, y = 362)

tk.Button(UI, text = "Add subassembly", width = 20, command = add_subassembly).place(x = 340, y = 360)

tk.Label(UI, text = "Note: x scale acts as the overall scale factor for the inserted file").place(x = 140, y = 304)


#add prism
tk.Label(UI, text = "Add prism", font = ("Ariel", 12)).place(x = 5, y = 405)
tk.Button(UI, text = "Add prism", width = 20, command = add_prism).place(x = 340, y = 525)

tk.Label(UI, text = "Name:").place(x = 5, y = 435)
prism_name_entry = tk.Entry(UI, width = 15)
prism_name_entry.place(x = 55, y = 437)

tk.Label(UI, text = "No. sides:").place(x = 5, y = 465)
side_count_entry = tk.Entry(UI, width = 13)
side_count_entry.place(x = 67, y = 467)

tk.Label(UI, text = "Radius:").place(x = 165, y = 465)
prism_radius_entry = tk.Entry(UI, width = 15)
prism_radius_entry.place(x = 215, y = 467)

tk.Label(UI, text = "Height:").place(x = 325, y = 465)
prism_height_entry = tk.Entry(UI, width = 15)
prism_height_entry.place(x = 375, y = 467)

tk.Label(UI, text = "Taper:").place(x = 5, y = 495)
taper_entry = tk.Entry(UI, width = 15)
taper_entry.place(x = 55, y = 497)

ishollow = tk.BooleanVar(UI)
tk.Label(UI, text = "Hollow:").place(x = 165, y = 495)
hollow_check = tk.Checkbutton(UI, variable = ishollow, command = check_hollow)
hollow_check.place(x = 215, y = 495)

tk.Label(UI, text = "Wall thickness:").place(x = 285, y = 495)
wall_thickness_entry = tk.Entry(UI, width = 15)
wall_thickness_entry.place(x = 375, y = 497)
 


#add dish
tk.Label(UI, text = "Add dish", font = ("Ariel", 12)).place(x = 5, y = 540)
tk.Button(UI, text = "Add dish", width = 20, command = add_dish).place(x = 340, y = 630)

tk.Label(UI, text = "Name:").place(x = 5, y = 570)
dish_name_entry = tk.Entry(UI, width = 15)
dish_name_entry.place(x = 55, y = 572)

tk.Label(UI, text = "Radius:").place(x = 5, y = 600)
dish_radius_entry = tk.Entry(UI, width = 15)
dish_radius_entry.place(x = 55, y = 602) 

tk.Label(UI, text = "Height:").place(x = 165, y = 600)
dish_height_entry = tk.Entry(UI, width = 15)
dish_height_entry.place(x = 215, y = 602)

tk.Label(UI, text = "Segements:").place(x = 5, y = 630)
dish_segments_entry = tk.Entry(UI, width = 12)
dish_segments_entry.place(x = 74, y = 632)

tk.Label(UI, text = "Fidelity:").place(x = 165, y = 630)
dish_fidelity_entry = tk.Entry(UI, width = 15)
dish_fidelity_entry.place(x = 215, y = 632)

#render options
render_config = tk.IntVar(UI, 3)
vertecies_select = tk.Radiobutton(UI, text = "Vertecies", value = 1, variable = render_config)
vertecies_select.place(x = 5, y = 680)
edges_selected = tk.Radiobutton(UI, text = "Edges", value = 2, variable = render_config)
edges_selected.place(x = 5, y = 700)
faces_selected = tk.Radiobutton(UI, text = "Faces", value = 3, variable = render_config)
faces_selected.place(x = 5, y = 720)


next_frame()

tk.mainloop()
UI.mainloop()
