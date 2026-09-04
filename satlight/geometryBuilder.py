from math import cos, sin, pi, sqrt
import os

folderpath = ''
filepath = ''
material_file = ''

object_count = 0
vertex_count = 0
texture_count = 0
global_rotation = [0,0,0]
filename = ''


def setup(folder, name, x_rotation = 0, y_rotation = 0, z_rotation = 0):
    #initialises save files
    global filepath
    global folderpath
    global filename
    global material_file
    global global_rotation

    global vertex_count
    global texture_count
    global normal_count 

    vertex_count = 0
    texture_count = 0
    normal_count = 0
    
    folderpath = folder
    filename = name

    #initialises obj and mtl files
    filepath = folder + '\\' + name + '.obj'
    material_file = folder + '\\' + name + '.mtl'

    #creates obj
    #references mtl in obj file
    with open(filepath, "w") as file:
        file.write('mtllib ' + filename + '.mtl\n')

    #creates mtl
    open(material_file, "w")

    #sets global rotations
    global_rotation = x_rotation, y_rotation, z_rotation


def filewrite(datapoints, datacode):
    #writes data points to obj file - vertecies and normals
    global filename
    data_type = str(datacode) + " "
    data_list = ""

    with open(filepath, "a") as file:

        #write data
        for i in range(len(datapoints)):

            point = datapoints[i]
            datum = ""

            #formatting
            for item in point:
                if datacode == 'vn':
                    datum += str("{:.4f}".format(item)) + " "
                else:
                    datum += str("{:.6f}".format(item)) + " "

            data_list += data_type + datum + "\n"

        file.write(data_list)



def mtlwrite(material):
    #writes the material from materialLibrary to the mtl file
    global material_file

    material_info = ''
    lib = os.path.dirname(__file__) + "\\materialLibrary.txt"

    with open(lib) as file:
        lines = file.readlines()
        line_count = 0
        for line in lines:
            if line.find(material) != -1: #finds reference material
                material_info += "\n"
                for i in range(line_count, line_count + 9): #material info has nine lines
                    material_info += lines[i]
                break
            line_count += 1
    material_info += "\n"
    with open(material_file, "a") as matfile:
        matfile.writelines(material_info)
                

def facewrite(datapoints, datacode, textures, normals, material):
    #writes face data

    data_type = str(datacode) + " "
    data_list = ""

    with open(filepath, "a") as file:
        #adds material
        file.write("usemtl " + material + "\n")
        mtlwrite(material)

        #adds shading value
        file.write("s 1 \n")

        #write data
        for i in range(len(datapoints)):

            point = datapoints[i]
            texturepoint = textures[i]
            normalvec = normals[i]
            datum = ""
            #formating
            for item in point:
                datum += str(item) + "/" + str(texturepoint) + "/" + str(normalvec) + " "

            data_list += data_type + datum + "\n"

        file.write(data_list)


def add_feature(vertecies, texture_coords, texture_index, normals, normal_index, faces, name, material):
    #handles the file writing per object
    global object_count

    object_count += 1
    with open(filepath, "a") as file:
        file.write("o " + material + "\n")

    #inputs data and datatype
    filewrite(vertecies, "v")
    filewrite(normals, "vn")
    filewrite(texture_coords, "vt")
    facewrite(faces, "f", texture_index, normal_index, material)


def maxtrix_transformation(transformation, vector):
    #3x3 times 3x1 matrix multiplication
    output_vector = [0, 0, 0]

    output_vector[0] = transformation[0][0] * vector[0] + transformation[0][1] * vector[1] + transformation[0][2] * vector[2]
    output_vector[1] = transformation[1][0] * vector[0] + transformation[1][1] * vector[1] + transformation[1][2] * vector[2]
    output_vector[2] = transformation[2][0] * vector[0] + transformation[2][1] * vector[1] + transformation[2][2] * vector[2] 

    return output_vector


def x_rotation(theta, vector):
    #sets up transformation matrix for x rotations
    transform = [[1, 0, 0],
                [0, cos(2 * pi * theta / 360), -sin(2 * pi * theta / 360)],
                [0, sin(2 * pi * theta / 360), cos(2 * pi * theta / 360)]]

    rotated_vector = maxtrix_transformation(transform, vector)

    return rotated_vector


def y_rotation(theta, vector):
    #sets up transformation matrix for y rotations
    transform = [[cos(2 * pi * theta / 360), 0, sin(2 * pi * theta / 360)],
                [0, 1, 0],
                [-sin(2 * pi * theta / 360), 0, cos(2 * pi * theta / 360)]]

    rotated_vector = maxtrix_transformation(transform, vector)

    return rotated_vector


def z_rotation(theta, vector):
    #sets up transformation matrix for z rotations
    transform = [[cos(2 * pi * theta / 360), -sin(2 * pi * theta / 360), 0],
                [sin(2 * pi * theta / 360), cos(2 * pi * theta / 360), 0],
                [0, 0, 1]]

    rotated_vector = maxtrix_transformation(transform, vector)

    return rotated_vector


def calc_normals(coords1, coords2, coords3):
    #calculates the normal vector of the face bound by three coordinates
    vec1 = [0, 0, 0]
    vec2 = [0, 0, 0]
    
    for i in range(len(coords1)):
        vec1[i] = coords2[i] - coords1[i]
        vec2[i] = coords3[i] - coords1[i]

    normal = [1, 1, 1]

    x1 = vec1[0]
    y1 = vec1[1]
    z1 = vec1[2]

    x2 = vec2[0]
    y2 = vec2[1]
    z2 = vec2[2]
    
    #cross product
    normal[0] = y1 * z2 - y2 * z1
    normal[1] = x2 * z1 - x1 * z2
    normal[2] = x1 * y2 - x2 * y1

    normal_leng = sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    normal[0] = round(normal[0] / normal_leng, 5)
    normal[1] = round(normal[1] / normal_leng, 5)
    normal[2] = round(normal[2] / normal_leng, 5)
    
    return normal


def import_model(file, folder = "", scale = 1, x_angle = 0, y_angle = 0, z_angle = 0, x_offset = 0, y_offset = 0, z_offset = 0):
    #reads an object file
    #applies rotations and offsets
    #updates obj reference values
    #writes data to obj file
    global folderpath
    global filepath
    global vertex_count
    global texture_count
    global normal_count
    global global_rotation
    global material_file

    object_lines = []
    vertex_track = 0
    texture_track = 0
    normal_track = 0

    if folder == "":
        folder = folderpath

    target_filepath = folder + '\\' + file + '.obj'

    with open(target_filepath) as file:

        lines = file.readlines()

        for line in lines:
            #object name
            if line.find('o ') != -1:

                object_lines.append(line)

            #vertecies
            elif line.find('v ') != -1:

                space_count = 0
                digit = ""
                coords = []

                for char in line:

                    if char == ' ':

                        space_count += 1
                        try: #trys adding the point to coordiate
                            coords.append(float(digit))
                            digit = ""
                        except: #first character 
                            pass
                    elif space_count > 0: #character is part of coordinate
                        try:
                            digit += str(int(char))
                        except:
                            if char == '.':
                                digit += char
                            elif char == '-':
                                digit += char
                            elif char == 'e':
                                digit += char

                #scales
                x = float(coords[0]) * scale
                y = float(coords[1]) * scale
                try:
                    z = float(coords[2]) * scale
                except:
                    coords.append(float(digit))
                    z = float(coords[2]) * scale

                vertex = [x, y, z]

                #rotations
                if x_angle != 0:
                 vertex = x_rotation(x_angle, vertex)

                if y_angle != 0:
                    vertex = y_rotation(y_angle, vertex)

                if z_angle != 0:
                    vertex = z_rotation(z_angle, vertex)

                #translations
                vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

                #formating
                vertex_line = "v " + str("{:.6f}".format(vertex[0])) + " " + str("{:.6f}".format(vertex[1])) + " " + str("{:.6f}".format(vertex[2])) + " " + "\n"

                object_lines.append(vertex_line)

                vertex_track += 1

            #texture coordinates
            elif line.find('vt ') != -1:
               
                object_lines.append(line)
                texture_track += 1

            #normals
            elif line.find('vn ') != -1:
                
                space_count = 0
                digit = ""
                coords = []

                for char in line:

                    if char == ' ':

                        space_count += 1
                        try: #trys adding point to normal
                            coords.append(float(digit))
                            digit = ""
                        except: #first character
                            pass
                    elif space_count > 0: #character is part of normal
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
                try:
                    z = float(coords[2])
                except:
                    coords.append(float(digit))
                    z = float(coords[2])

                normal = [x, y, z]
                #rotations
                if x_angle != 0:
                    normal = x_rotation(x_angle, normal)

                if y_angle != 0:
                    normal = y_rotation(y_angle, normal)

                if z_angle != 0:
                    normal = z_rotation(z_angle, normal)

                # convert to unit vector
                normal_leng = sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
                normal[0] = round(normal[0] / normal_leng, 5)
                normal[1] = round(normal[1] / normal_leng, 5)
                normal[2] = round(normal[2] / normal_leng, 5)

                #formating
                vertex_line = "vn " + str("{:.4f}".format(normal[0])) + " " + str("{:.4f}".format(normal[1])) + " " + str("{:.4f}".format(normal[2])) + " " + "\n"

                object_lines.append(vertex_line)

                normal_track += 1

            #shading values
            elif line.find('s ') != -1:

                object_lines.append(line)

            #faces
            elif line.find('f ') != -1:
                
                index_location = 0
                face_line = ""
                number = ""

                for char in line:

                    if char == ' ' or char == '\n':
                        if index_location == 2: #third digit is the normal
                            number = str(int(number) + normal_count)
                            face_line += number
                            number = ''

                        index_location = 0

                    if char == '/':
                        if index_location == 0: #first digit is vertex
                            number = str(int(number) + vertex_count)
                        elif index_location == 1: #second digit is texture coordinate
                            try:
                                number = str(int(number) + texture_count)
                            except:
                                pass

                        face_line += number
                        number = ''
                        index_location += 1

                    try: #tracks numbers
                        number += str(int(char))
                        
                    except: #adds formatting back in
                        face_line += str(char)
                
                
                object_lines.append(face_line)

            #material references
            elif line.find("usemtl ") != -1:
                object_lines.append(line)
                material_name = line.split(" ", 1)[1]
                material_info = ''
                lib = target_filepath.split(".", 1)[0] + '.mtl'

                with open(lib) as libfile:
                    lines = libfile.readlines()
                    line_count = 0
                    for line in lines:
                        if line.find(material_name) != -1: #finds referenced material
                            material_info += '\n'
                            for i in range(line_count, line_count + 9): #materials have 9 lines of data
                                material_info += lines[i]
                            break
                        line_count += 1

                #writes material to mtl
                with open(material_file, 'a') as matfile:
                    matfile.writelines(material_info)



        #write data
        with open(filepath, "a") as writefile:

            for i in range(len(object_lines)):

                writefile.write(object_lines[i])

    #update references
    vertex_count += vertex_track
    texture_count += texture_track
    normal_count += normal_track

    

#calculate geometry
def prism_geometry(name, side_count, radius, height, taper = 1, is_hollow = False, wall_thickness = 0, material = "MLI", x_scale = 1, y_scale = 1, z_scale = 1, x_angle = 0, y_angle = 0, z_angle = 0, x_offset = 0, y_offset = 0, z_offset = 0):
    #adds a prism
    #generates vertecies 
    #calculates normals
    #writes data to obj
    global vertex_count
    global texture_count
    global normal_count
    global global_rotation

    
    theta = 360 / side_count
    vertecies_list = []
    face_list = []
    top_face = []
    bottom_face = []
    current_face = []
    normal_list = []
    normal_reference = []
    texture_list = []
    texture_reference = []
    texture_map_val = []

    #prisms with even number of sides are square to axes, odd number of sides have top vertex
    if side_count % 2 == 0:
        current_angle = theta / 2
    else:
        current_angle = 0

    if is_hollow == True:
        revolutions = 2
    else:
        revolutions = 1
    
    for i in range(side_count * revolutions + (2 - revolutions)):


        #calculates vertex - used to create a pair of vertecies separated by the prisms height
        if i == (side_count * revolutions):
            #centre of top and bottom faces
            x = 0
            y = 0
            z = (height / 2) * z_scale
        elif current_angle >= 360:
            #interior of hollow prism
            x = (radius - wall_thickness) * cos(2 * pi * current_angle / 360) * x_scale
            y = (radius - wall_thickness) * sin(2 * pi * current_angle / 360) * y_scale
            z = (height / 2) * z_scale
        else:
            #exterior 
            x = radius * cos(2 * pi * current_angle / 360) * x_scale
            y = radius * sin(2 * pi * current_angle / 360) * y_scale
            z = (height / 2) * z_scale

        #top vertex
        vertex = [x / taper, y / taper, z]
        
        # part rotations
        if x_angle != 0:
            vertex = x_rotation(x_angle, vertex)

        if y_angle != 0:
            vertex = y_rotation(y_angle, vertex)

        if z_angle != 0:
            vertex = z_rotation(z_angle, vertex)
        
        #translation
        vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

        #global rotations
        if global_rotation[0] != 0:
            vertex = x_rotation(global_rotation[0], vertex)

        if global_rotation[1] != 0:
            vertex = y_rotation(global_rotation[1], vertex)

        if global_rotation[2] != 0:
            vertex = z_rotation(global_rotation[2], vertex)

        #add to list
        vertecies_list.extend([vertex])
        top_face.append((2 * i + 1) + vertex_count)

        #bottom vertex
        vertex = [x, y, -z]
        
        #part rotations
        if x_angle != 0:
            vertex = x_rotation(x_angle, vertex)

        if y_angle != 0:
            vertex = y_rotation(y_angle, vertex)

        if z_angle != 0:
            vertex = z_rotation(z_angle, vertex)

        #translations
        vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

        #global rotations
        if global_rotation[0] != 0:
            vertex = x_rotation(global_rotation[0], vertex)

        if global_rotation[1] != 0:
            vertex = y_rotation(global_rotation[1], vertex)

        if global_rotation[2] != 0:
            vertex = z_rotation(global_rotation[2], vertex)

        #add to list
        vertecies_list.extend([vertex])
        bottom_face.append((2 * i + 2) + vertex_count)


        #add rectangular faces
        if i != 0 and i != side_count:
        #side faces
            j = 2 * i
            if current_angle <= 360:
                current_face = (j - 1 + vertex_count, j + vertex_count, j + 2 + vertex_count, j + 1 + vertex_count)
            else:
                current_face = (j + 1 + vertex_count, j + 2 + vertex_count, j + vertex_count, j - 1 + vertex_count)
            face_list.extend([current_face])
            current_face = []

            #normals
            normal_vector = calc_normals(vertecies_list[-4], vertecies_list[-3], vertecies_list[-2])
            if current_angle > 360:
                normal_vector = [-normal_vector[0], -normal_vector[1], -normal_vector[2]]
            normal_list.extend([normal_vector])
            normal_reference.append(i + normal_count)


        #adds exterior sides final rectangle
        if i == side_count - 1:
            j = 2 * i
            current_face = (j + 1 + vertex_count, j + 2 + vertex_count, 2 + vertex_count, 1 + vertex_count)
            face_list.extend([current_face])

            #normals
            normal_vector = calc_normals(vertecies_list[-2], vertecies_list[-1], vertecies_list[0])
            normal_list.extend([normal_vector])
            normal_reference.append(i + 1 + normal_count)

        #adds interior sides final rectangle
        if i == 2 * side_count - 1:
            j = 2 * i
            current_face = (2 * side_count + 1 + vertex_count, 2 * side_count + 2 + vertex_count, j + 2 + vertex_count, j + 1 + vertex_count)
            face_list.extend([current_face])

            #normals
            normal_vector = calc_normals(vertecies_list[-2], vertecies_list[-1], vertecies_list[2 * side_count])
            if current_angle > 360:
                normal_vector = [-normal_vector[0], -normal_vector[1], -normal_vector[2]]
            normal_list.extend([normal_vector])
            normal_reference.append(i + 1 + normal_count)
        
        current_angle += theta


    #adds top and bottom faces
    if is_hollow == False:

        #top

        #top normal
        normal_vector = calc_normals(vertecies_list[-2], vertecies_list[-4], vertecies_list[0])
        normal_vector = [normal_vector[0], normal_vector[1], normal_vector[2]]
        normal_list.extend([normal_vector])

        #top faces
        for i in range(side_count - 1):

          normal_reference.append(side_count + 1 + normal_count) 
          current_face = (top_face[-1], top_face[i], top_face[i + 1])
          face_list.extend([current_face])
        #final top face
        normal_reference.append(side_count + 1 + normal_count)  
        current_face = (top_face[-1], top_face[-2], top_face[0])
        face_list.extend([current_face])


        #bottom

        #bottom normal
        normal_vector = calc_normals(vertecies_list[-1], vertecies_list[-3], vertecies_list[1])
        normal_vector = [-normal_vector[0], -normal_vector[1], -normal_vector[2]]
        normal_list.extend([normal_vector])

        #bottom faces
        for i in range(side_count - 1):

          normal_reference.append(side_count + 2 + normal_count) 
          current_face = (bottom_face[i + 1], bottom_face[i], bottom_face[-1])
          face_list.extend([current_face])

        #final bottom face
        normal_reference.append(side_count + 2 + normal_count)
        current_face = (bottom_face[0], bottom_face[-2], bottom_face[-1])
        face_list.extend([current_face])


    #top and bottom faces for hollow prisms
    if is_hollow == True:
         
        #top

        #top normal
        normal_vector = calc_normals(vertecies_list[0], vertecies_list[2], vertecies_list[2 * side_count])
        normal_vector = [normal_vector[0], normal_vector[1], normal_vector[2]]
        normal_list.extend([normal_vector])

        #top faces
        for i in range(side_count - 1):

          normal_reference.append(2 * side_count + 1 + normal_count) 
          current_face = (top_face[i], top_face[i + 1], top_face[side_count + 1 + i], top_face[side_count + i])
          face_list.extend([current_face])

        #final top face
        normal_reference.append(2 * side_count + 1 + normal_count)  
        current_face = (top_face[side_count - 1], top_face[0], top_face[side_count], top_face[2 * side_count - 1])
        face_list.extend([current_face])

        #bottom

        #bottom normal
        normal_vector = calc_normals(vertecies_list[1], vertecies_list[3], vertecies_list[2 * side_count + 1])
        normal_vector = [-normal_vector[0], -normal_vector[1], -normal_vector[2]]
        normal_list.extend([normal_vector])

        #bottom faces
        for i in range(side_count - 1):

          normal_reference.append(2 * side_count + 2 + normal_count) 
          current_face = (bottom_face[side_count + i], bottom_face[side_count + 1 + i], bottom_face[i + 1], bottom_face[i])
          face_list.extend([current_face])

        #final bottom face
        normal_reference.append(2 * side_count + 2 + normal_count)
        current_face = (bottom_face[2 * side_count - 1], bottom_face[side_count], bottom_face[0], bottom_face[side_count - 1])
        face_list.extend([current_face])

    #textures - generates placeholder texture coordinates
    for i in range(side_count * 2 + side_count * revolutions):

        texture_map_val = (1, 1)
        texture_list.extend([texture_map_val])
        texture_reference.append(i + 1 + texture_count)

    #updates references
    if is_hollow ==False:
        vertex_count += 2 * side_count + 2
        texture_count += side_count * 3
        normal_count += side_count + 2
    else:
        vertex_count += 4 * side_count
        texture_count += 4 * side_count
        normal_count += 2 * side_count + 2
    
    
    #data to be written
    add_feature(vertecies_list, texture_list, texture_reference, normal_list, normal_reference, face_list, name, material)



def parabola_geometry(name, radius, height, segments, fidelity = 36, material = "White_Paint", x_angle = 0, y_angle = 0, z_angle = 0, x_offset = 0, y_offset = 0, z_offset = 0):
    #generates parabolas
    #calculates parabola based on radius and height
    #calculates vertecies
    #applies rotations and translations
    #updates references
    #writes data to obj
    global vertex_count
    global normal_count
    global texture_count
    global global_rotation

    a_coeff = height / (radius ** 2)
    segment_spacing = radius / segments
    fidelity = fidelity
    theta = 360/fidelity
    current_angle = 0
    face_list = []
    vertex_list = []
    current_face = []
    normal_list = []
    normal_reference = []
    texture_list = []
    texture_reference = []
    half_height = 0.01


    #interior side

    #centre point
    x = 0
    y = 0
    z = 0 + half_height

    vertex = (x, y, z)

    #rotation
    if x_angle != 0:
        vertex = x_rotation(x_angle, vertex)

    if y_angle != 0:
        vertex = y_rotation(y_angle, vertex)

    if z_angle != 0:
        vertex = z_rotation(z_angle, vertex)
  
    #translation
    vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

    #global rotation
    if global_rotation[0] != 0:
        vertex = x_rotation(global_rotation[0], vertex)

    if global_rotation[1] != 0:
        vertex = y_rotation(global_rotation[1], vertex)

    if global_rotation[2] != 0:
        vertex = z_rotation(global_rotation[2], vertex)

    vertex_list.extend([vertex])


    for i in range(segments): #radius split into segments
        for j in range(fidelity): #angle split into fidelity
            #calculates vertex
            x = (i + 1) * segment_spacing * cos(2 * pi * current_angle / 360)
            y = (i + 1) * segment_spacing * sin(2 * pi * current_angle / 360)
            z = a_coeff * ((i + 1) * segment_spacing) ** 2 + half_height

            vertex = (x, y, z)

            #rotation
            if x_angle != 0:
                vertex = x_rotation(x_angle, vertex)

            if y_angle != 0:
                vertex = y_rotation(y_angle, vertex)

            if z_angle != 0:
                vertex = z_rotation(z_angle, vertex)
        
            #translation
            vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

            #global rotation
            if global_rotation[0] != 0:
                vertex = x_rotation(global_rotation[0], vertex)

            if global_rotation[1] != 0:
                vertex = y_rotation(global_rotation[1], vertex)

            if global_rotation[2] != 0:
                vertex = z_rotation(global_rotation[2], vertex)

            vertex_list.extend([vertex])

            #face list
            if i == 0 and j > 0: #adds first ring of triangular faces
                current_face = (j + 1 + vertex_count, j + 2 + vertex_count, 1 + vertex_count)
                face_list.extend([current_face])

                #normal
                normal_vector = calc_normals(vertex_list[j + 1], vertex_list[0], vertex_list[j])
                normal_vector = [normal_vector[0], normal_vector[1], normal_vector[2]]
                normal_list.extend([normal_vector])
                normal_reference.append(j + normal_count)

                #final triangle
                if j == fidelity - 1:
                    current_face = (j + 2 + vertex_count, 2 + vertex_count, 1 + vertex_count)
                    face_list.extend([current_face])

                    #normal
                    normal_vector = calc_normals(vertex_list[0], vertex_list[1], vertex_list[j + 1])
                    normal_vector = [-normal_vector[0], -normal_vector[1], -normal_vector[2]]
                    normal_list.extend([normal_vector])
                    normal_reference.append(j + 1 + normal_count)

            if i > 0 and j > 0: #adds all other faces - rectangles
                current_face = (fidelity * i + j + 1 + vertex_count, fidelity * i + j + 2 + vertex_count, fidelity * (i - 1) + j + 2 + vertex_count, fidelity * (i - 1) + j + 1 + vertex_count)
                face_list.extend([current_face])

                #normals
                normal_vector = calc_normals(vertex_list[fidelity * (i - 1) + j], vertex_list[fidelity * (i - 1) + j + 1], vertex_list[fidelity * i + j + 1])
                normal_vector = [-normal_vector[0], -normal_vector[1], -normal_vector[2]]
                normal_list.extend([normal_vector])
                normal_reference.append(fidelity * i + j + normal_count)

                #final rectangle
                if j == fidelity - 1:
                    current_face = (fidelity * i + j + 2 + vertex_count, fidelity * i + 2 + vertex_count, fidelity * (i - 1) + 2 + vertex_count, fidelity * (i - 1) + j + 2 + vertex_count)
                    face_list.extend([current_face])

                    #normal
                    normal_vector = calc_normals(vertex_list[fidelity * (i - 1) + j + 1], vertex_list[fidelity * (i - 1) + 1], vertex_list[fidelity * i + j + 1])
                    normal_vector = [-normal_vector[0], -normal_vector[1], -normal_vector[2]]
                    normal_list.extend([normal_vector])
                    normal_reference.append(fidelity * i + j + 1+ normal_count)

            current_angle += theta
         

    #exterior side

    #centre point
    x = 0
    y = 0
    z = 0 - half_height

    vertex = (x, y, z)
    #rotation
    if x_angle != 0:
        vertex = x_rotation(x_angle, vertex)

    if y_angle != 0:
        vertex = y_rotation(y_angle, vertex)

    if z_angle != 0:
        vertex = z_rotation(z_angle, vertex)
        
    #translation
    vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

    #global rotation
    if global_rotation[0] != 0:
        vertex = x_rotation(global_rotation[0], vertex)

    if global_rotation[1] != 0:
        vertex = y_rotation(global_rotation[1], vertex)

    if global_rotation[2] != 0:
        vertex = z_rotation(global_rotation[2], vertex)

    vertex_list.extend([vertex])


    #corrects indexing for bottom face
    parabola_vertex_indexing = fidelity * segments + 1
    parabola_normal_indexing = fidelity * segments


    for i in range(segments): #radius split into segments
        for j in range(fidelity): #angle split into fidelity
            #calculates vertex
            x = (i + 1) * segment_spacing * cos(2 * pi * current_angle / 360)
            y = (i + 1) * segment_spacing * sin(2 * pi * current_angle / 360)
            z = a_coeff * ((i + 1) * segment_spacing) ** 2 - half_height

            vertex = (x, y, z)

            #rotation
            if x_angle != 0:
                vertex = x_rotation(x_angle, vertex)

            if y_angle != 0:
                vertex = y_rotation(y_angle, vertex)

            if z_angle != 0:
                vertex = z_rotation(z_angle, vertex)
        
            #translation
            vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

            #global rotation
            if global_rotation[0] != 0:
                vertex = x_rotation(global_rotation[0], vertex)

            if global_rotation[1] != 0:
                vertex = y_rotation(global_rotation[1], vertex)

            if global_rotation[2] != 0:
                vertex = z_rotation(global_rotation[2], vertex)

            vertex_list.extend([vertex])

            #face list
            if i == 0 and j > 0: #adds first ring of triangular faces
                current_face = (1 + vertex_count + parabola_vertex_indexing, j + 2 + vertex_count + parabola_vertex_indexing, j + 1 + vertex_count + parabola_vertex_indexing)
                face_list.extend([current_face])

                #normal
                normal_vector = calc_normals(vertex_list[j + 1], vertex_list[0], vertex_list[j])
                normal_vector = [-normal_vector[0], -normal_vector[1], -normal_vector[2]]
                normal_list.extend([normal_vector])
                normal_reference.append(j + normal_count + parabola_normal_indexing)

                #final triangle
                if j == fidelity - 1:
                    current_face = (1 + vertex_count + parabola_vertex_indexing, 2 + vertex_count + parabola_vertex_indexing, j + 2 + vertex_count + parabola_vertex_indexing)
                    face_list.extend([current_face])

                    #normal
                    normal_vector = calc_normals(vertex_list[0], vertex_list[1], vertex_list[j + 1])
                    normal_vector = [normal_vector[0], normal_vector[1], normal_vector[2]]
                    normal_list.extend([normal_vector])
                    normal_reference.append(j + 1 + normal_count + parabola_normal_indexing)

            if i > 0 and j > 0: #adds all other faces - rectangles
                current_face = (fidelity * (i - 1) + j + 1 + vertex_count + parabola_vertex_indexing, fidelity * (i - 1) + j + 2 + vertex_count + parabola_vertex_indexing, fidelity * i + j + 2 + vertex_count + parabola_vertex_indexing, fidelity * i + j + 1 + vertex_count + parabola_vertex_indexing)
                face_list.extend([current_face])

                #normals
                normal_vector = calc_normals(vertex_list[fidelity * (i - 1) + j], vertex_list[fidelity * (i - 1) + j + 1], vertex_list[fidelity * i + j + 1])
                normal_vector = [normal_vector[0], normal_vector[1], normal_vector[2]]
                normal_list.extend([normal_vector])
                normal_reference.append(fidelity * i + j + normal_count + parabola_normal_indexing)

                #final rectangle
                if j == fidelity - 1:
                    current_face = (fidelity * (i - 1) + j + 2 + vertex_count + parabola_vertex_indexing, fidelity * (i - 1) + 2 + vertex_count + parabola_vertex_indexing, fidelity * i + 2 + vertex_count + parabola_vertex_indexing, fidelity * i + j + 2 + vertex_count + parabola_vertex_indexing)
                    face_list.extend([current_face])

                    #normal
                    normal_vector = calc_normals(vertex_list[fidelity * (i - 1) + j + 1], vertex_list[fidelity * (i - 1) + 1], vertex_list[fidelity * i + j + 1])
                    normal_vector = [normal_vector[0], normal_vector[1], normal_vector[2]]
                    normal_list.extend([normal_vector])
                    normal_reference.append(fidelity * i + j + 1 + normal_count + parabola_normal_indexing)

            current_angle += theta


    #outer edge
    #finds first outer vertex for top and bottom parabolas
    reference_zero_vertex_top = (segments - 1) * fidelity + 1 + 1
    reference_zero_vertex_bottom = (segments - 1) * fidelity + 1 + segments * fidelity + 1 + 1

    #adds faces
    for i in range(fidelity - 1):
        #faces
        current_face = (reference_zero_vertex_top + i + vertex_count, reference_zero_vertex_bottom + i + vertex_count, reference_zero_vertex_bottom + i + 1 + vertex_count, reference_zero_vertex_top + i + 1 + vertex_count)
        face_list.extend([current_face])

        #normals
        normal_vector = calc_normals(vertex_list[reference_zero_vertex_top + i - 1], vertex_list[reference_zero_vertex_bottom + i - 1], vertex_list[reference_zero_vertex_bottom + i + 1 - 1])
        normal_vector = [normal_vector[0], normal_vector[1], normal_vector[2]]
        normal_list.extend([normal_vector])
        normal_reference.append(parabola_normal_indexing * 2 + i + 1)

    #final rectangle
    #face
    current_face = (reference_zero_vertex_top + fidelity - 1 + vertex_count, reference_zero_vertex_bottom + fidelity - 1 + vertex_count, reference_zero_vertex_bottom + vertex_count, reference_zero_vertex_top + vertex_count)
    face_list.extend([current_face])

    #normal
    normal_vector = calc_normals(vertex_list[reference_zero_vertex_top - 1], vertex_list[reference_zero_vertex_bottom - 1], vertex_list[reference_zero_vertex_bottom + fidelity - 1 - 1])
    normal_vector = [-normal_vector[0], -normal_vector[1], -normal_vector[2]]
    normal_list.extend([normal_vector])
    normal_reference.append(parabola_normal_indexing * 2 + i + 2)


    #textures - generates placeholder texture coordinates
    for i in range(2 * fidelity * segments + fidelity):

        texture_map_val = (1, 1)
        texture_list.extend([texture_map_val])
        texture_reference.append(i + 1 + texture_count)

    #updates references
    vertex_count += 2 * (fidelity * segments + 1)
    normal_count += 2 * (fidelity * segments) + fidelity
    texture_count += 2 * (fidelity * segments) + fidelity

    #data to be written
    add_feature(vertex_list, texture_list, texture_reference, normal_list, normal_reference, face_list, name, material)



setup(r"A:\Uni work\2nd year internship\Satlight program", "write test")
import_model("Surface solar panel 1x1", r"A:\Uni work\2nd year internship\Satlight program\Files for git\Components\Solar panels\High detail")
