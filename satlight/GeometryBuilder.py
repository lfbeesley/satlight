from math import *

folderpath = ''
filepath = ''

object_count = 0
vertex_count = 0
texture_count = 0
normal_count = 0
global_rotation = [0,0,0]


def setup(folder, name, x_rotation = 0, y_rotation = 0, z_rotation = 0):

    global filepath
    global folderpath

    global vertex_count
    global texture_count
    global normal_count 

    vertex_count = 0
    texture_count = 0
    normal_count = 0
    
    folderpath = folder

    filepath = folder + name + '.obj'
    open (filepath, "w")

    global_rotation = x_rotation, y_rotation, z_rotation


def filewrite(datapoints, datacode):

    data_type = str(datacode) + " "
    data_list = ""

    with open(filepath, "a") as file:

        #write data
        for i in range(len(datapoints)):

            point = datapoints[i]
            datum = ""

            for item in point:
                if datacode == 'vn':
                    datum += str("{:.4f}".format(item)) + " "
                else:
                    datum += str("{:.6f}".format(item)) + " "

            data_list += data_type + datum + "\n"

        file.write(data_list)



def facewrite(datapoints, datacode, textures, normals):

    data_type = str(datacode) + " "
    data_list = ""

    with open(filepath, "a") as file:

        file.write("s 1 \n")
        #write data
        for i in range(len(datapoints)):

            point = datapoints[i]
            texturepoint = textures[i]
            normalvec = normals[i]
            datum = ""

            for item in point:
                datum += str(item) + "/" + str(texturepoint) + "/" + str(normalvec) + " "

            data_list += data_type + datum + "\n"

        file.write(data_list)


def add_feature(vertecies, texture_coords, texture_index, normals, normal_index, faces, name):

    global object_count

    object_count += 1
    with open(filepath, "a") as file:
        file.write("o " + name + "\n")

    filewrite(vertecies, "v")
    filewrite(normals, "vn")
    filewrite(texture_coords, "vt")
    facewrite(faces, "f", texture_index, normal_index)


def maxtrix_transformation(transformation, vector):

    output_vector = [0, 0, 0]

    output_vector[0] = transformation[0][0] * vector[0] + transformation[1][0] * vector[1] + transformation[2][0] * vector[2]
    output_vector[1] = transformation[0][1] * vector[0] + transformation[1][1] * vector[1] + transformation[2][1] * vector[2]
    output_vector[2] = transformation[0][2] * vector[0] + transformation[1][2] * vector[1] + transformation[2][2] * vector[2] 

    return output_vector


def x_rotation(theta, vector):

    transform = [[1, 0, 0],
                [0, cos(2 * pi * theta / 360), -sin(2 * pi * theta / 360)],
                [0, sin(2 * pi * theta / 360), cos(2 * pi * theta / 360)]]

    rotated_vector = maxtrix_transformation(transform, vector)

    return rotated_vector


def y_rotation(theta, vector):

    transform = [[cos(2 * pi * theta / 360), 0, sin(2 * pi * theta / 360)],
                [0, 1, 0],
                [-sin(2 * pi * theta / 360), 0, cos(2 * pi * theta / 360)]]

    rotated_vector = maxtrix_transformation(transform, vector)

    return rotated_vector


def z_rotation(theta, vector):

    transform = [[cos(2 * pi * theta / 360), -sin(2 * pi * theta / 360), 0],
                [sin(2 * pi * theta / 360), cos(2 * pi * theta / 360), 0],
                [0, 0, 1]]

    rotated_vector = maxtrix_transformation(transform, vector)

    return rotated_vector


def calc_normals(coords1, coords2, coords3):

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

    global folderpath
    global filepath
    global vertex_count
    global texture_count
    global normal_count
    global global_rotation

    object_lines = []
    vertex_track = 0
    texture_track = 0
    normal_track = 0

    if folder == "":
        folder = folderpath

    target_filepath = folder + file + '.obj'

    with open(target_filepath) as file:

        lines = file.readlines()

        for line in lines:
            
            if line.find('o') != -1:

                object_lines.append(line)



            elif line.find('v ') != -1:

                space_count = 0
                digit = ""
                coords = []

                for char in line:

                    if char == ' ':

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
                            if char == '-':
                                digit += char
                            if char == 'e':
                                digit += char

                x = float(coords[0]) * scale
                y = float(coords[1]) * scale
                z = float(coords[2]) * scale

                vertex = [x, y, z]

                if x_angle != 0:
                 vertex = x_rotation(x_angle, vertex)

                if y_angle != 0:
                    vertex = y_rotation(y_angle, vertex)

                if z_angle != 0:
                    vertex = z_rotation(z_angle, vertex)

                vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

                vertex_line = "v " + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + " " + "\n"

                object_lines.append(vertex_line)

                vertex_track += 1



            elif line.find('vt') != -1:
               
                object_lines.append(line)
                texture_track += 1

            elif line.find('vn') != -1:
                
                space_count = 0
                digit = ""
                coords = []

                for char in line:

                    if char == ' ':

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
                            if char == '-':
                                digit += char
                            if char == 'e':
                                digit += char

                x = float(coords[0])
                y = float(coords[1])
                z = float(coords[2])

                normal = [x, y, z]

                if x_angle != 0:
                    normal = x_rotation(x_angle, normal)

                if y_angle != 0:
                    normal = y_rotation(y_angle, normal)

                if z_angle != 0:
                    normal = z_rotation(z_angle, normal)


                normal_leng = sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
                normal[0] = round(normal[0] / normal_leng, 5)
                normal[1] = round(normal[1] / normal_leng, 5)
                normal[2] = round(normal[2] / normal_leng, 5)


                vertex_line = "vn " + str(normal[0]) + " " + str(normal[1]) + " " + str(normal[2]) + " " + "\n"

                object_lines.append(vertex_line)

                normal_track += 1

            elif line.find('s') != -1:

                object_lines.append(line)

            elif line.find('f') != -1:
                
                index_location = 0
                face_line = ""
                number = ""

                for char in line:

                    if char == ' ':
                        if index_location == 2:
                            number = str(int(number) + normal_count)
                            face_line += number
                            number = ''

                        index_location = 0

                    if char == '/':
                        if index_location == 0:
                            number = str(int(number) + vertex_count)
                        elif index_location == 1:
                            number = str(int(number) + texture_count)

                        face_line += number
                        number = ''
                        index_location += 1

                    try:
                        a = int(char)
                        number += char
                        
                    except:
                        face_line += str(char)
                
                
                object_lines.append(face_line)

        with open(filepath, "a") as writefile:

            for i in range(len(object_lines)):

                writefile.write(object_lines[i])


    vertex_count += vertex_track
    texture_count += texture_track
    normal_count += normal_track

    

#calculate geometry
def prism_geometry(name, side_count, radius, height, taper = 1, is_hollow = False, wall_thickness = 0, x_scale = 1, y_scale = 1, z_scale = 1, x_angle = 0, y_angle = 0, z_angle = 0, x_offset = 0, y_offset = 0, z_offset = 0):

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

    if side_count % 2 == 0:
        current_angle = theta / 2
    else:
        current_angle = 0

    if is_hollow == True:
        revolutions = 2
    else:
        revolutions = 1
    
    for i in range(side_count * revolutions):

        if current_angle >= 360:
            x = (radius - wall_thickness) * cos(2 * pi * current_angle / 360) * x_scale
            y = (radius - wall_thickness) * sin(2 * pi * current_angle / 360) * y_scale
            z = (height / 2) * z_scale
        else:
            x = radius * cos(2 * pi * current_angle / 360) * x_scale
            y = radius * sin(2 * pi * current_angle / 360) * y_scale
            z = (height / 2) * z_scale

        #top vertex
        vertex = [x / taper, y / taper, z]
        
        #part rotations
        if x_angle != 0:
            vertex = x_rotation(x_angle, vertex)

        if y_angle != 0:
            vertex = y_rotation(y_angle, vertex)

        if z_angle != 0:
            vertex = z_rotation(z_angle, vertex)
        
        #part translation
        vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

        #model rotations
        if global_rotation[0] != 0:
            vertex = x_rotation(global_rotation[0], vertex)

        if global_rotation[1] != 0:
            vertex = y_rotation(global_rotation[1], vertex)

        if global_rotation[2] != 0:
            vertex = z_rotation(global_rotation[2], vertex)


        vertecies_list.extend([vertex])
        top_face.append((2 * i + 1) + vertex_count)

        #bottom vertex
        vertex = [x, y, -z]
        
        if x_angle != 0:
            vertex = x_rotation(x_angle, vertex)

        if y_angle != 0:
            vertex = y_rotation(y_angle, vertex)

        if z_angle != 0:
            vertex = z_rotation(z_angle, vertex)

        vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

        if global_rotation[0] != 0:
            vertex = x_rotation(global_rotation[0], vertex)

        if global_rotation[1] != 0:
            vertex = y_rotation(global_rotation[1], vertex)

        if global_rotation[2] != 0:
            vertex = z_rotation(global_rotation[2], vertex)


        vertecies_list.extend([vertex])
        bottom_face.append((2 * i + 2) + vertex_count)


        #add rectangular faces
        if i != 0 and i != side_count:

            j = 2 * i

            current_face = (j - 1 + vertex_count, j + vertex_count, j + 2 + vertex_count, j + 1 + vertex_count)
            face_list.extend([current_face])
            current_face = []

            #normals
            normal_vector = calc_normals(vertecies_list[-4], vertecies_list[-3], vertecies_list[-2])
            normal_list.extend([normal_vector])
            normal_reference.append(i + normal_count)


        #adds exterior final rectangle
        if i == side_count - 1:
            j = 2 * i
            current_face = (j + 1 + vertex_count, j + 2 + vertex_count, 2 + vertex_count, 1 + vertex_count)
            face_list.extend([current_face])

            normal_vector = calc_normals(vertecies_list[-2], vertecies_list[-1], vertecies_list[0])
            normal_list.extend([normal_vector])
            normal_reference.append(i + 1 + normal_count)

        #adds interior final rectangle
        if i == 2 * side_count - 1:
            j = 2 * i
            current_face = (j + 1 + vertex_count, j + 2 + vertex_count, 2 * side_count + 2 + vertex_count, 2 * side_count + 1 + vertex_count)
            face_list.extend([current_face])

            normal_vector = calc_normals(vertecies_list[-2], vertecies_list[-1], vertecies_list[2 * side_count])
            normal_list.extend([normal_vector])
            normal_reference.append(i + 1 + normal_count)
        
        current_angle += theta


    #adds top and bottom faces
    if is_hollow == False:

        normal_vector = calc_normals(vertecies_list[-2], vertecies_list[-4], vertecies_list[0])
        normal_list.extend([normal_vector])
        normal_reference.append(side_count + 1 + normal_count)


        normal_vector = calc_normals(vertecies_list[-1], vertecies_list[-3], vertecies_list[1])
        normal_list.extend([normal_vector])
        normal_reference.append(side_count + 2 + normal_count)


        face_list.extend([top_face])
        face_list.extend([bottom_face])

    if is_hollow == True:
         
        #top
        normal_vector = calc_normals(vertecies_list[0], vertecies_list[2], vertecies_list[2 * side_count])
        normal_list.extend([normal_vector])

        for i in range(side_count - 1):

          normal_reference.append(2 * side_count + 1 + normal_count) 
          current_face = (top_face[i], top_face[i + 1], top_face[side_count + 1 + i], top_face[side_count + i])
          face_list.extend([current_face])

        normal_reference.append(2 * side_count + 1 + normal_count)  
        current_face = (top_face[side_count - 1], top_face[0], top_face[side_count], top_face[2 * side_count - 1])
        face_list.extend([current_face])

        #bottom
        normal_vector = calc_normals(vertecies_list[1], vertecies_list[3], vertecies_list[2 * side_count + 1])
        normal_list.extend([normal_vector])

        for i in range(side_count - 1):

          normal_reference.append(2 * side_count + 2 + normal_count) 
          current_face = (bottom_face[i], bottom_face[i + 1], bottom_face[side_count + 1 + i], bottom_face[side_count + i])
          face_list.extend([current_face])

        normal_reference.append(2 * side_count + 2 + normal_count)
        current_face = (bottom_face[side_count - 1], bottom_face[0], bottom_face[side_count], bottom_face[2 * side_count - 1])
        face_list.extend([current_face])


    for i in range(revolutions * revolutions * side_count + 2):

        texture_map_val = (1/(i+1)**2, 1/(i+1))
        texture_list.extend([texture_map_val])
        texture_reference.append(i + 1 + texture_count)

    
    if is_hollow ==False:
        vertex_count += 2 * side_count
        texture_count += side_count + 2
        normal_count += side_count + 2
    else:
        vertex_count += 4 * side_count
        texture_count += 4 * side_count
        normal_count += 2 * side_count + 2
    
    

    add_feature(vertecies_list, texture_list, texture_reference, normal_list, normal_reference, face_list, name)

def parabola_geometry(name, radius, height, segments, fidelity = 100, x_angle = 0, y_angle = 0, z_angle = 0, x_offset = 0, y_offset = 0, z_offset = 0):
    
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

    x = 0 + x_offset
    y = 0 + y_offset
    z = 0 + z_offset

    vertex = (x, y, z)
    vertex_list.extend([vertex])

    for i in range(segments):
        for j in range(fidelity):

            x = (i + 1) * segment_spacing * cos(2 * pi * current_angle / 360)
            y = (i + 1) * segment_spacing * sin(2 * pi * current_angle / 360)
            z = a_coeff * ((i + 1) * segment_spacing) ** 2

            vertex = (x, y, z)

            #rotation and translation
            if x_angle != 0:
                vertex = x_rotation(x_angle, vertex)

            if y_angle != 0:
                vertex = y_rotation(y_angle, vertex)

            if z_angle != 0:
                vertex = z_rotation(z_angle, vertex)
        

            vertex = (vertex[0] + x_offset, vertex[1] + y_offset, vertex[2] + z_offset)

            if global_rotation[0] != 0:
                vertex = x_rotation(global_rotation[0], vertex)

            if global_rotation[1] != 0:
                vertex = y_rotation(global_rotation[1], vertex)

            if global_rotation[2] != 0:
                vertex = z_rotation(global_rotation[2], vertex)

            vertex_list.extend([vertex])

            #face list
            if i == 0 and j > 0:
                current_face = (1 + vertex_count, j + 1 + vertex_count, j + 2 + vertex_count)
                face_list.extend([current_face])

                normal_vector = calc_normals(vertex_list[0], vertex_list[j], vertex_list[j + 1])
                normal_list.extend([normal_vector])
                normal_reference.append(j + normal_count)

                if j == fidelity - 1:
                    current_face = (1 + vertex_count, 2 + vertex_count, j + 2 + vertex_count)
                    face_list.extend([current_face])

                    normal_vector = calc_normals(vertex_list[0], vertex_list[1], vertex_list[j + 1])
                    normal_list.extend([normal_vector])
                    normal_reference.append(j + 1 + normal_count)

            if i > 0 and j > 0:
                current_face = (fidelity * (i - 1) + j + 1 + vertex_count, fidelity * (i - 1) + j + 2 + vertex_count, fidelity * i + j + 2 + vertex_count, fidelity * i + j + 1 + vertex_count)
                face_list.extend([current_face])

                normal_vector = calc_normals(vertex_list[fidelity * (i - 1) + j], vertex_list[fidelity * (i - 1) + j + 1], vertex_list[fidelity * i + j + 1])
                normal_list.extend([normal_vector])
                normal_reference.append(fidelity * i + j + normal_count)
                 
                if j == fidelity - 1:
                    current_face = (fidelity * (i - 1) + j + 2 + vertex_count, fidelity * (i - 1) + 2 + vertex_count, fidelity * i + 2 + vertex_count, fidelity * i + j + 2 + vertex_count)
                    face_list.extend([current_face])

                    normal_vector = calc_normals(vertex_list[fidelity * (i - 1) + j + 1], vertex_list[fidelity * (i - 1) + 1], vertex_list[fidelity * i + j + 1])
                    normal_list.extend([normal_vector])
                    normal_reference.append(fidelity * i + j + 1+ normal_count)

            current_angle += theta
            
    #textures
    for i in range(fidelity * segments):

        texture_map_val = (1/(i+1)**2, 1/(i+1))
        texture_list.extend([texture_map_val])
        texture_reference.append(i + 1 + texture_count)

    vertex_count += fidelity * segments + 1
    normal_count += fidelity * segments
    texture_count += fidelity * segments

    add_feature(vertex_list, texture_list, texture_reference, normal_list, normal_reference, face_list, name)
