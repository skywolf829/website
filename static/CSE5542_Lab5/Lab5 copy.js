

///////////////////////////////////////////////////////////////////////
//
//      Skylar Wurster
//      Lab 5 - Realtime Graphics
//      Autumn 2019
//
///////////////////////////////////////////////////////////////////////
/**
*   A class to hold anything that a game object would need to remember
*   location, rotation, scale, etc.
*/
class GameObject{
    constructor(){
        this.name = "";
        this.localScale = [1, 1, 1];
        this.rotation = [0, 0, 0];
        this.translate = [0, 0, 0];     
        this.global_position = [0, 0, 0];
        this.boundingCircle = 0.05;
        this.ambient = [0.0, 0.0, 0.0];
        this.diffuse = [0.0, 0.0, 0.0];
        this.specular = [1.0, 1.0, 1.0];
        this.shininess = 100.0;
        this.textureOffset = [0, 0];
        this.textureScale = [1, 1];
        // variables for the tree structure of the scene
        this.root = null;
        this.children = [];
        // Has variables to identify which vertices in the vbo are its own to manipulate/draw   
        this.vertex_normal_index = [];
        this.vertex_texture_index = [];
        this.vertex_element_index = [];

        this.vertex_start_index = 0;
        this.vertex_num_indices = 0;
        this.color_start_index = 0;
        this.color_num_indices = 0;
        this.normal_start_index = 0;
        this.normal_num_indices = 0;
        this.texturecoord_start_index = 0;
        this.texturecoord_num_indicies = 0;
        // The type of gameObject this is, default is LINES, but can be POINTS or TRIANGLES as well
        this.poly_type = "TRIANGLES";
        this.drawType = "TRIANGLES";
        this.texture = null;
        this.cubeMap = null;
        this.normalMap = null;        
        this.useTexture = false;
        this.useNormalMap = false;
        this.useReflectionMapping = false;
        this.useLighting = true;
        this.enabled = true;
    }

    /**
    *   Creates the MMatrix by matrix multiplication 
    */
    get_m_matrix(){        
        this.mMatrix = mat4.create();
        this.mMatrix = mat4.identity(this.mMatrix);
        this.mMatrix = mat4.translate(this.mMatrix, this.translate);    
        this.mMatrix = mat4.scale(this.mMatrix, this.localScale);    
        this.mMatrix = mat4.rotate(this.mMatrix, degToRad(this.rotation[0]), [1, 0, 0]);
        this.mMatrix = mat4.rotate(this.mMatrix, degToRad(this.rotation[1]), [0, 1, 0]);
        this.mMatrix = mat4.rotate(this.mMatrix, degToRad(this.rotation[2]), [0, 0, 1]);
        return this.mMatrix;
    }
    /**
    *   Simply checks to see if the click is within some distance of the global position, calculated in draw.
    *   Does not use vertex positions.
    */
    in_bounding_box(NDC_X, NDC_Y){
        return Math.sqrt(Math.pow(this.global_position[0] - NDC_X, 2) + Math.pow(this.global_position[1] - NDC_Y, 2)) < this.boundingCircle * this.localScale[0];
    }    
    global_scale(){
        if(this.root != null){
            return [this.localScale[0] * this.root.global_scale()[0],
            this.localScale[1] * this.root.global_scale()[1],
            this.localScale[2] * this.root.global_scale()[2],
            ];
        }
        else return this.localScale;
    }
    get_color(){
        return [vbo_colors[this.color_start_index*4],
                vbo_colors[this.color_start_index*4+1],
                vbo_colors[this.color_start_index*4+2]];
    }
    set_color(color){
        for(var i = this.color_start_index*4; i < this.color_start_index*4 + this.color_num_indices*4; i+=4){
            vbo_colors[i] = color[0];
            vbo_colors[i+1] = color[1];
            vbo_colors[i+2] = color[2];
        }        
    }
    /**
    *   Draws the gameobject with world_matrix being the parent matrix to account for 
    */
    draw(world_matrix){
        var final_matrix = mat4.create();
        final_matrix = mat4.multiply(world_matrix, this.get_m_matrix(), final_matrix);
        // Save global position for other uses
        this.global_position = mat4.multiplyVec4(final_matrix, [0, 0, 0, 1]);
        // Create the normal matrix for the object
        var nMatrix = mat4.create();
        nMatrix = mat4.identity(nMatrix);        
        nMatrix = mat4.multiply(nMatrix,vMatrix);        
        nMatrix = mat4.multiply(nMatrix,final_matrix );        
        nMatrix = mat4.inverse(nMatrix);
        nMatrix = mat4.transpose(nMatrix);     
        
        // Draw based on the type of gameobject it is
        if(this.vertex_num_indices == 0){
            return;
        }
        else{
            gl.uniformMatrix4fv(shaderProgram.mMatrixUniform, false, final_matrix);
            gl.uniformMatrix4fv(shaderProgram.nMatrixUniform, false, nMatrix);
            gl.uniformMatrix4fv(shaderProgram.v2wMatrix, false, v2wMatrix);
            gl.uniform3f(shaderProgram.materialAmbient, this.ambient[0], this.ambient[1], this.ambient[2]);
            gl.uniform3f(shaderProgram.materialDiffuse, this.diffuse[0], this.diffuse[1], this.diffuse[2]);
            gl.uniform3f(shaderProgram.materialSpecular, this.specular[0], this.specular[1], this.specular[2]);
            gl.uniform1f(shaderProgram.materialShininess, this.shininess);
            gl.uniform2f(shaderProgram.textureOffset, this.textureOffset[0], this.textureOffset[1]);
            gl.uniform2f(shaderProgram.textureScale, this.textureScale[0], this.textureScale[1]);
            

            gl.bindBuffer(gl.ARRAY_BUFFER, VertexPositionBuffer);
            gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 0, 0);

            gl.bindBuffer(gl.ARRAY_BUFFER, VertexNormalBuffer);
            gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 0, 0);

            gl.bindBuffer(gl.ARRAY_BUFFER, VertexColorBuffer);
            gl.vertexAttribPointer(shaderProgram.vertexColorAttribute, 4, gl.FLOAT, false, 0, 0);

            gl.bindBuffer(gl.ARRAY_BUFFER, VertexTextureCoordsBuffer);
            gl.vertexAttribPointer(shaderProgram.vertexTextureCoordinateAttribute, 2, gl.FLOAT, false, 0, 0);

            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, VertexIndexBuffer);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(this.vertex_element_index), gl.STATIC_DRAW);

            
            gl.uniform1i(shaderProgram.use_textureUniform, this.useTexture);    
            gl.uniform1i(shaderProgram.use_normalMapUniform, this.useNormalMap);                       
            gl.uniform1i(shaderProgram.use_reflectionMappingUniform, this.useReflectionMapping);
            gl.uniform1i(shaderProgram.use_lightingUniform, this.useLighting);
          
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.texture);
            gl.uniform1i(shaderProgram.textureUniform, 0);
        
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_CUBE_MAP, this.cubeMap);
            gl.uniform1i(shaderProgram.cubeMapTextureUniform, 1);

            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, this.normalMap);
            gl.uniform1i(shaderProgram.normalUniform, 2);
            


            if(this.drawType == "TRIANGLES"){
                gl.drawElements(gl.TRIANGLES, this.vertex_element_index.length, gl.UNSIGNED_SHORT, 0);
            }
            else if(this.drawType == "LINES"){
                gl.drawArrays(gl.LINES,this.vertex_start_index, this.vertex_num_indices);
            }
            else if(this.drawType == "POINTS"){
                gl.drawArrays(gl.POINTS, this.vertex_start_index, this.vertex_num_indices);
            }
        }
    }
}
function init_texture(g, src){
    g.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, g.texture);  
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
        new Uint8Array([0, 0, 255, 255]));
    
    g.texture.image = new Image();
    g.texture.image.crossOrigin = "anonymous";
    
    g.texture.image.addEventListener('load', function(){
        gl.bindTexture(gl.TEXTURE_2D, g.texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB,gl.UNSIGNED_BYTE, g.texture.image);

        // WebGL1 has different requirements for power of 2 images
        // vs non power of 2 images so check if the image is a
        // power of 2 in both dimensions.
        if (isPowerOf2(g.texture.image.width) && isPowerOf2(g.texture.image.height)) {
            // Yes, it's a power of 2. Generate mips.
            gl.generateMipmap(gl.TEXTURE_2D);
        } else {
            // No, it's not a power of 2. Turn off mips and set
            // wrapping to clamp to edge
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        }
        g.useTexture = true;        
    
        CreateBuffers();
        draw_scene();
    });
    
    g.texture.image.src = src;    
}
function init_normalMap(g, src){
    g.normalMap = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, g.normalMap);  
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
        new Uint8Array([0, 0, 255, 255]));
    
    g.normalMap.image = new Image();
    g.normalMap.image.crossOrigin = "anonymous";
    
    g.normalMap.image.addEventListener('load', function(){
        gl.bindTexture(gl.TEXTURE_2D, g.normalMap);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB,gl.UNSIGNED_BYTE, g.normalMap.image);

        // WebGL1 has different requirements for power of 2 images
        // vs non power of 2 images so check if the image is a
        // power of 2 in both dimensions.
        if (isPowerOf2(g.normalMap.image.width) && isPowerOf2(g.normalMap.image.height)) {
            // Yes, it's a power of 2. Generate mips.
            gl.generateMipmap(gl.TEXTURE_2D);
        } else {
            // No, it's not a power of 2. Turn off mips and set
            // wrapping to clamp to edge
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        }
        g.useNormalMap = true;        
    
        CreateBuffers();
        draw_scene();
    });
    
    g.normalMap.image.src = src;    
}
function init_cubeMap(g, src){
    g.cubeMap = gl.createTexture();    
    g.cubeMap.frontImage = new Image();
    g.cubeMap.frontImage.crossOrigin = "anonymous";
    g.cubeMap.backImage = new Image();
    g.cubeMap.backImage.crossOrigin = "anonymous";
    g.cubeMap.leftImage = new Image();
    g.cubeMap.leftImage.crossOrigin = "anonymous";
    g.cubeMap.rightImage = new Image();
    g.cubeMap.rightImage.crossOrigin = "anonymous";
    g.cubeMap.topImage = new Image();
    g.cubeMap.topImage.crossOrigin = "anonymous";
    g.cubeMap.botImage = new Image();
    g.cubeMap.botImage.crossOrigin = "anonymous";
    
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, g.cubeMap);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.REPEAT); 
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR); 

    g.cubeMap.frontImage.addEventListener('load', function(){
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, g.cubeMap);
        gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_Z, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
            g.cubeMap.frontImage);
        g.useReflectionMapping = true;    
        CreateBuffers();
        draw_scene();
    });
    g.cubeMap.backImage.addEventListener('load', function(){
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, g.cubeMap);
        gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
            g.cubeMap.backImage);
        g.useReflectionMapping = true;    
        CreateBuffers();
        draw_scene();
    });
    g.cubeMap.rightImage.addEventListener('load', function(){
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, g.cubeMap);
        gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
            g.cubeMap.rightImage);
        g.useReflectionMapping = true;    
        CreateBuffers();
        draw_scene();
    });
    g.cubeMap.leftImage.addEventListener('load', function(){
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, g.cubeMap);
        gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_X, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
            g.cubeMap.leftImage);
        g.useReflectionMapping = true;    
        CreateBuffers();
        draw_scene();
    });
    g.cubeMap.topImage.addEventListener('load', function(){
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, g.cubeMap);
        gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_Y, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
            g.cubeMap.topImage);
        g.useReflectionMapping = true;    
        CreateBuffers();
        draw_scene();
    });
    g.cubeMap.botImage.addEventListener('load', function(){
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, g.cubeMap);
        gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
            g.cubeMap.botImage);
        g.useReflectionMapping = true;    
        CreateBuffers();
        draw_scene();
    });
    
    g.cubeMap.frontImage.src = src+"_front.jpg";
    g.cubeMap.backImage.src = src+"_back.jpg";
    g.cubeMap.leftImage.src = src+"_left.jpg";
    g.cubeMap.rightImage.src = src+"_right.jpg";
    g.cubeMap.topImage.src = src+"_top.jpg";
    g.cubeMap.botImage.src = src+"_bottom.jpg";
}
var gl;  // the graphics context (gc) 
var shaderProgram;  // the shader program 

//viewport info 
var vp_minX, vp_maxX, vp_minY, vp_maxY, vp_width, vp_height; 

var VertexPositionBuffer;
var VertexColorBuffer;
var VertexNormalBuffer;
var VertexIndexBuffer;
var VertexTextureCoordsBuffer;

var vbo_vertices = []; 
var vbo_colors = []; 
var vbo_normals = [];
var vbo_texcoords = [];

var pMatrix = mat4.create();  //projection matrix
var vMatrix = mat4.create(); // view matrix
var v2wMatrix = mat4.create(); //view to world matrix
var cam_pos, cam_look_at, cam_pitch_yaw_roll;
cam_pos = vec3.create();
cam_look_at = vec3.create();
cam_pitch_yaw_roll = vec3.create();
cam_pos = vec3.set([0, 0, -50], cam_pos);
cam_look_at  = vec3.set([0, 0, 0], cam_look_at);
cam_pitch_yaw_roll = vec3.set([0, 0, 0], cam_pitch_yaw_roll);

var lightPosition = [0.0, 30.0, 0.0];
var lightAmbient = [1.0,1.0, 1.0];
var lightDiffuse = [1.0, 1.0, 1.0];
var lightSpecular = [1.0, 1.0, 1.0];

/**
*   Simply converts degrees to radians for OpenGL
*/
function degToRad(degrees) {
    return degrees * Math.PI / 180;
}

function add_vertex(vert){
    vbo_vertices.push(vert[0]);
    vbo_vertices.push(vert[1]);
    vbo_vertices.push(vert[2]);
}
function add_normal(norm){
    vbo_normals.push(norm[0]);
    vbo_normals.push(norm[1]);
    vbo_normals.push(norm[2]);
}
function add_color(c, times){
    for(var i = 0; i < times; i++){
        vbo_colors.push(c[0]);
        vbo_colors.push(c[1]);
        vbo_colors.push(c[2]);
        vbo_colors.push(1.0);
    }
}
function add_texturecoord(tex_coord){
    vbo_texcoords.push(tex_coord[0])
    vbo_texcoords.push(tex_coord[1]);
}
function isPowerOf2(value) {
    return (value & (value - 1)) == 0;
}
function sin(a){
    return Math.sin(a);
}
function cos(a){
    return Math.cos(a);
}

/**
 * 
 * Creates and returns a Klein bottle shadow in 3D
 * 
 */
function create_klein_bottle(xSteps, ySteps, color){
    var g = new GameObject();
    g.poly_type = "TRIANGLES";
    g.vertex_start_index = vbo_vertices.length / 3;
    g.color_start_index = vbo_colors.length / 4;
    g.texturecoord_start_index = vbo_texcoords.length / 2;
    var pi = Math.PI;
    
    for(var i = 0; i <= xSteps; i++){
        for(var j = 0; j <= ySteps; j++){
            var x = -1.0 + 2*(i/xSteps);
            var y = 0 + 1*(j/ySteps);
            var ax = (4+sin(pi*x)*cos(pi*y)-sin(2*pi*x)*sin(pi*y))*cos(2*pi*y);
            var ay = (4+sin(pi*x)*cos(pi*y)-sin(2*pi*x)*sin(pi*y))*sin(2*pi*y);
            var az = sin(pi*x)*sin(pi*y)+sin(2*pi*x)*cos(pi*y);
            var dpdx_x = ((4+pi*cos(pi*x)*cos(pi*y)) - 2*pi*cos(2*pi*x)*sin(pi*y))*cos(2*pi*y);
            var dpdx_y = ((4+pi*cos(pi*x)*cos(pi*y)) - 2*pi*cos(2*pi*x)*sin(pi*y))*sin(2*pi*y)
            var dpdx_z = pi*cos(pi*x)*sin(pi*y) + 2*pi*cos(2*pi*x)*cos(pi*y);
            var dpdy_x = (4+sin(pi*x)*-pi*sin(pi*y)-sin(2*pi*x)*pi*cos(pi*y))*cos(2*pi*y) +
            (4+sin(pi*x)*cos(pi*y)-sin(2*pi*x)*sin(pi*y))*cos(2*pi*y)*-2*pi*sin(2*pi*y);
            var dpdy_y = (4+sin(pi*x)*-pi*sin(pi*y)-sin(2*pi*x)*pi*cos(pi*y))*sin(2*pi*y) +
            (4+sin(pi*x)*cos(pi*y)-sin(2*pi*x)*sin(pi*y))*cos(2*pi*y)*2*pi*cos(2*pi*y)
            var dpdy_z = sin(pi*x)*pi*cos(pi*y)+sin(2*pi*x)*-pi*sin(pi*y)
            var dpdx = vec3.create([dpdx_x, dpdx_y, dpdx_z]);
            var dpdy = vec3.create([dpdy_x, dpdy_y, dpdy_z]);
            var normal = vec3.create();
            normal = vec3.cross(dpdx, dpdy, normal);
            add_vertex([ax, ay, az]);
            add_texturecoord([(x+1)/2.0, y]);
            add_normal(normal);
            add_color([1, 1, 1]);
        }
    }
    for(var i = 0; i < xSteps; i++){
        for(var j = 0; j < ySteps; j++){
            var ind1 = (i*xSteps)+j;
            var ind2 = ((i+1)*xSteps)+j;
            var ind3 = j == ySteps - 1 ? ((i+1)*xSteps) : ((i+1)*xSteps)+j+1;
            var ind4 = j == ySteps - 1 ? (i*xSteps) : (i*xSteps)+j+1;
            g.vertex_element_index.push(g.vertex_start_index + ind1);
            g.vertex_element_index.push(g.vertex_start_index + ind2);
            g.vertex_element_index.push(g.vertex_start_index + ind3);
            
            g.vertex_element_index.push(g.vertex_start_index + ind1);
            g.vertex_element_index.push(g.vertex_start_index + ind3);
            g.vertex_element_index.push(g.vertex_start_index + ind4);
        }
    }
    g.vertex_num_indices = (vbo_vertices.length / 3) -  g.vertex_start_index;
    g.color_num_indices = (vbo_colors.length / 4) - g.color_start_index;
    g.texturecoord_num_indicies = (vbo_texcoords.length / 2) - g.texturecoord_start_index;
    g.diffuse = color;
    g.ambient = [color[0] * 0.3, color[1] * 0.3, color[2] * 0.3]; 
    return g;

}

/**
 * Creates a returns a new sphere gameobject.
 * radius is the radius of the sphere
 * num_slices is the number of points per height cross section (more gives 
 *      finer granularity)
 * stacks is the number of slices inside the sphere, discounting the top and 
 *      bottom points
 * color is the color of the sphere verts
 *
 */
function create_sphere(xSteps, ySteps, color){ 
    var g = new GameObject();
    g.poly_type = "TRIANGLES";
    g.drawType = "TRIANGLES";
    g.vertex_start_index = vbo_vertices.length / 3;
    g.color_start_index = vbo_colors.length / 4;
    g.texturecoord_start_index = vbo_texcoords.length / 2;
    var pi = Math.PI;
    
    for(var i = 0; i <= xSteps; i++){
        for(var j = 0; j <= ySteps; j++){
            var x = -1.0 + 2*(i/xSteps);
            var y = -0.5 + 1*(j/ySteps);
            var ax = cos(pi*y)*cos(pi*x);
            var ay = cos(pi*y)*sin(pi*x);
            var az = sin(pi*y);
            add_vertex([ax, ay, az]);
            add_texturecoord([(x+1)/2.0, y+0.5]);
            add_normal([ax, ay, az]);
            add_color(color);
        }
    }
    for(var i = 0; i <= xSteps; i++){
        for(var j = 0; j <= ySteps; j++){
            var ind1 = (i*xSteps)+j;
            var ind2 = ((i+1)*xSteps)+j;
            var ind3 = j == ySteps - 1 ? ((i+1)*xSteps) : ((i+1)*xSteps)+j+1;
            var ind4 = j == ySteps - 1 ? (i*xSteps) : (i*xSteps)+j+1;
            g.vertex_element_index.push(g.vertex_start_index + ind1);
            g.vertex_element_index.push(g.vertex_start_index + ind2);
            g.vertex_element_index.push(g.vertex_start_index + ind3);
            
            g.vertex_element_index.push(g.vertex_start_index + ind1);
            g.vertex_element_index.push(g.vertex_start_index + ind3);
            g.vertex_element_index.push(g.vertex_start_index + ind4);
        }
    }

    g.vertex_num_indices = (vbo_vertices.length / 3) -  g.vertex_start_index;
    g.color_num_indices = (vbo_colors.length / 4) - g.color_start_index;
    g.texturecoord_num_indicies = (vbo_texcoords.length / 2) - g.texturecoord_start_index;
    g.diffuse = color;
    g.ambient = [color[0] * 0.3, color[1] * 0.3, color[2] * 0.3]; 
    return g;
}

/**
 * Creates a returns a new cylinder gameobject.
 * base radius is the radius of the bottom of the cylinder
 * top radius is the radius of the top of the cylinder
 * height is the height of the cylinder
 * num_slices is the number of slices on each height level of the cylinder
 * stacks is the number of vertical divisions of points between the top and
 *      bottom levels
 * color is the color of the cylinder verts
 *
 */
function create_cylinder(base_radius, top_radius, height, num_slices, stacks, color){
    var g = new GameObject();
    g.poly_type = "TRIANGLES";
    g.vertex_start_index = vbo_vertices.length / 3;
    g.color_start_index = vbo_colors.length / 4;

    // add the bottom/top cylinder points
    add_vertex([0, -height/2, 0]);
    add_texturecoord([0.5, 0]);
    add_normal([0, -1, 0]);
    add_vertex([0, height/2, 0]);    
    add_texturecoord([0.5, 1.0]);
    add_normal([0, 1, 0]);
    // create bottom ring
    for(var i = 0; i < num_slices; i++){
        var rads = (i / num_slices) * 2 * Math.PI;
        var x = Math.cos(rads) * base_radius;
        var z = Math.sin(rads) * base_radius;
        var y = -height / 2;
        var v = [x, y, z];
        add_vertex(v);
        add_normal([x, 0, z]);
        add_texturecoord([i / num_slices, 0]);
    }
    // create the bottom circle
    for(var i = 0; i < num_slices; i++){     
        g.vertex_element_index.push(g.vertex_start_index + (0));
        g.vertex_element_index.push(g.vertex_start_index + (i+2));
        g.vertex_element_index.push(g.vertex_start_index + (i == num_slices - 1 ? 2 : i+3));
    }
    //create each layer going up
    for(var i = 0; i < stacks; i++){
        for(var j = 0; j < num_slices; j++){
            var currentR = base_radius - ((i+1) / (stacks+1)) * (base_radius - top_radius);
            var rads = (j / num_slices) * 2 * Math.PI;
            var x = Math.cos(rads) * currentR;
            var z = Math.sin(rads) * currentR;
            var y = (-height/2) + (i+1)/(stacks+1) * (height);
            var v = [x, y, z];
            add_vertex(v);            
            add_texturecoord([j / num_slices, i / stacks]);
            add_normal([x, (base_radius - top_radius) / height, z]);
        }
        // and link it to the preceding layer
        var start = (i+1) * num_slices + 2;
        
        for(var j = 0; j < num_slices; j++){            
            var ind1 = start+j;
            var ind2 = j == num_slices - 1 ? start : start+j+1;
            var ind3 = (start+j-num_slices);
            var ind4 = j == num_slices - 1 ? start-num_slices : start+j-num_slices+1;
            g.vertex_element_index.push(g.vertex_start_index + ind1);
            g.vertex_element_index.push(g.vertex_start_index + ind2);            
            g.vertex_element_index.push(g.vertex_start_index + ind3);
            
            g.vertex_element_index.push(g.vertex_start_index + ind2);
            g.vertex_element_index.push(g.vertex_start_index + ind4);
            g.vertex_element_index.push(g.vertex_start_index + ind3);
        }

    }

    //create the top ring   
    start = (i+1) * num_slices + 2;
    for(var i = 0; i < num_slices; i++){
        var rads = (i / num_slices) * 2 * Math.PI;
        var x = Math.cos(rads) * top_radius;
        var z = Math.sin(rads) * top_radius;
        var y = height / 2;
        add_vertex([x, y, z]);       
        add_texturecoord([i / num_slices, 1.0]);
        add_normal([x, 0, z]);
    }
    
    // connect top point to top ring
    for(var i = 0; i < num_slices; i++){
        g.vertex_element_index.push(g.vertex_start_index + (1));
        g.vertex_element_index.push(g.vertex_start_index + (i+start));
        g.vertex_element_index.push(g.vertex_start_index + (i == num_slices - 1 ? start : i+1+start));
    }

    // connect the top ring to last layer
    for(var i = 0; i < num_slices; i++){
        var ind1 = start+i;
        var ind2 = i == num_slices - 1 ? start : start+i+1;
        var ind3 = start+i-num_slices;
        var ind4 = i == num_slices - 1 ? start-num_slices : start+i-num_slices+1
        
        g.vertex_element_index.push(g.vertex_start_index + ind1);
        g.vertex_element_index.push(g.vertex_start_index + ind2);            
        g.vertex_element_index.push(g.vertex_start_index + ind3);
        
        g.vertex_element_index.push(g.vertex_start_index + ind2);
        g.vertex_element_index.push(g.vertex_start_index + ind4);
        g.vertex_element_index.push(g.vertex_start_index + ind3);
    }
    
    
    add_color(color, (vbo_vertices.length / 3) -  g.vertex_start_index);

    g.vertex_num_indices = (vbo_vertices.length / 3) -  g.vertex_start_index;
    g.color_num_indices = (vbo_colors.length / 4) -  g.color_start_index;
    g.boundingCircle = height/2;
    g.diffuse = color;
    g.ambient = [color[0] * 0.3, color[1] * 0.3, color[2] * 0.3];
    return g;
}

/**
 * Creates a cube with edge length size
 */
function create_cube(size, color){
    var g = new GameObject();
    g.poly_type = "TRIANGLES";
    g.vertex_start_index = vbo_vertices.length / 3;
    g.color_start_index = vbo_colors.length / 4;
    g.texturecoord_start_index = vbo_texcoords.length / 2;
    g.vertex_num_indices = 36;
    g.color_num_indices = 36;
    g.texturecoord_num_indicies = 36;

    var vert_1 = [-size/2, -size/2, -size/2];   
    var vert_2 = [size/2, -size/2, -size/2]; 
    var vert_3 = [size/2, size/2, -size/2]; 
    var vert_4 = [-size/2, size/2, -size/2]; 
    var vert_5 = [-size/2, -size/2, size/2]; 
    var vert_6 = [size/2, -size/2, size/2]; 
    var vert_7 = [size/2, size/2, size/2]; 
    var vert_8 = [-size/2, size/2, size/2]; 
    
    var norm_1 = [0, 0, -1];
    var norm_2 = [1, 0, 0];
    var norm_3 = [0, 0, 1];
    var norm_4 = [-1, 0, 0];
    var norm_5 = [0, 1, 0];
    var norm_6 = [0, -1, 0];

    //front face
    add_vertex(vert_1);
    add_texturecoord([1.0, 0.66]);
    add_normal(norm_1);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_2);
    add_texturecoord([0.75, 0.66]);
    add_normal(norm_1);    
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_3);
    add_texturecoord([0.75, 0.33]);
    add_normal(norm_1);    
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    add_vertex(vert_1);
    add_texturecoord([1.0, 0.66]);
    add_normal(norm_1);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_3);
    add_texturecoord([0.75, 0.33]);
    add_normal(norm_1);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_4);
    add_texturecoord([1.0, 0.33]);
    add_normal(norm_1);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    //right
    add_vertex(vert_2);
    add_texturecoord([0.75, 0.66]);
    add_normal(norm_2);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_6);
    add_texturecoord([0.5, 0.66]);
    add_normal(norm_2);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_7);
    add_texturecoord([0.5, 0.33]);
    add_normal(norm_2);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    add_vertex(vert_2);
    add_texturecoord([0.75, 0.66]);
    add_normal(norm_2);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_7);
    add_texturecoord([0.5, 0.33]);
    add_normal(norm_2);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_3);
    add_texturecoord([0.75, 0.33]);
    add_normal(norm_2);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    //back
    add_vertex(vert_6);
    add_texturecoord([0.5, 0.66]);
    add_normal(norm_3);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_5);
    add_texturecoord([0.25, 0.66]);
    add_normal(norm_3);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_8);
    add_texturecoord([0.25, 0.33]);
    add_normal(norm_3);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    add_vertex(vert_6);
    add_texturecoord([0.5, 0.66]);
    add_normal(norm_3);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_8);
    add_texturecoord([0.25, 0.33]);
    add_normal(norm_3);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_7);
    add_texturecoord([0.5, 0.33]);
    add_normal(norm_3);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    //left
    add_vertex(vert_5);
    add_texturecoord([0.25, 0.66]);
    add_normal(norm_4);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_1);
    add_texturecoord([0.0, 0.66]);
    add_normal(norm_4);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_4);
    add_texturecoord([0.0, 0.33]);
    add_normal(norm_4);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    add_vertex(vert_5);
    add_texturecoord([0.25, 0.66]);
    add_normal(norm_4);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_4);
    add_texturecoord([0.0, 0.33]);
    add_normal(norm_4);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_8);
    add_texturecoord([0.25, 0.33]);
    add_normal(norm_4);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    //top
    add_vertex(vert_4);
    add_texturecoord([0.25, 0.0]);
    add_normal(norm_5);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_3);
    add_texturecoord([0.5, 0.0]);
    add_normal(norm_5);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_7);
    add_texturecoord([0.5, 0.33]);
    add_normal(norm_5);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    add_vertex(vert_4);
    add_texturecoord([0.25, 0.0]);
    add_normal(norm_5);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_7);
    add_texturecoord([0.5, 0.33]);
    add_normal(norm_5);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_8);
    add_texturecoord([0.25, 0.33]);
    add_normal(norm_5);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    //bottom
    add_vertex(vert_1);
    add_texturecoord([0.25, 1.0]);
    add_normal(norm_6);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_2);
    add_texturecoord([0.5, 1.0]);
    add_normal(norm_6);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_6);
    add_texturecoord([0.5, 0.66]);
    add_normal(norm_6);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    add_vertex(vert_1);
    add_texturecoord([0.25, 1.0]);
    add_normal(norm_6);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_6);
    add_texturecoord([0.5, 0.66]);
    add_normal(norm_6);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    add_vertex(vert_5);
    add_texturecoord([0.25, 0.66]);
    add_normal(norm_6);
    g.vertex_element_index.push((vbo_vertices.length / 3) - 1);
    
    add_color(color, 36);
    
    g.boundingCircle = size/2;
    g.diffuse = color;
    g.ambient = [color[0] * 0.3, color[1] * 0.3, color[2] * 0.3];
    return g;
}

// Scene graph is the root node for the scene
var scene_graph = new GameObject();
var lightGO;
scene_graph.name = "Scene Graph";
// The variable that controls which object is being changed 
var selected_gameObject = null;
// Used for rotation tracking
var mouse_down = false;
// Both used for rotation tracking
var start_NDC_X = 0;
var start_NDC_Y = 0;

var base, chest, rightArm, rightForearm, rightHand, leftArm, leftForearm, leftHand, head;
/** 
* Was used when attempting the bonus points for
* getting the width of some DOM element.
*/
var getStyle = function (e, styleName) {
    var styleValue = "";
    if(document.defaultView && document.defaultView.getComputedStyle) {
        styleValue = document.defaultView.getComputedStyle(e, "").getPropertyValue(styleName);
    }
    else if(e.currentStyle) {
        styleName = styleName.replace(/\-(\w)/g, function (strMatch, p1) {
            return p1.toUpperCase();
        });
        styleValue = e.currentStyle[styleName];
    }

    return styleValue;
}
//////////// Init OpenGL Context etc. ///////////////

/**
* Initializes the GL canvas.
*/
function initGL(canvas) {
    try {
        gl = canvas.getContext("experimental-webgl");
        gl.canvasWidth = document.getElementById("lab5-canvas").width; //window.innerWidth;
        gl.canvasHeight = document.getElementById("lab5-canvas").height; //window.innerHeight;
    } catch (e) {
    }
    if (!gl) {
        alert("Could not initialize WebGL");
    }
}

function loadOBJfile(location, translate, rotate, scale, color, useTexture=null, useCubemap=null){
    var xhr = new XMLHttpRequest(); 
    xhr.open("GET", location); 
    xhr.responseType = "text";
    //force the HTTP response, response-type header to be blob

    xhr.onload = function(e) { 
        var contents = e.target.response;
        var g = new GameObject();
        g.poly_type = "TRIANGLES";
        g.vertex_start_index = vbo_vertices.length / 3;
        g.normal_start_index = vbo_normals.length / 3;
        g.color_start_index = vbo_colors.length / 4;
        g.texturecoord_start_index = vbo_texcoords.length / 2;
        
        tokens = tokenize(contents);
        for(var i = 0; i < tokens.length; ){
            if(tokens[i] == "v"){
                add_vertex([Number(tokens[i+1]), Number(tokens[i+2]), Number(tokens[i+3])]);
                i = i + 4;
            }
            else if(tokens[i] == "vn"){
                add_normal([Number(tokens[i+1]), Number(tokens[i+2]), Number(tokens[i+3])]);
                i = i + 4;
            }
            else if(tokens[i] == "vt"){
                add_texturecoord([Number(tokens[i+1]), Number(tokens[i+2])]);
                i = i + 3;
                if(tokens[i+3] == 0){
                    i = i + 1;
                }
            }
            else if(tokens[i] == "f"){
                g.vertex_element_index.push(g.vertex_start_index - 1 + Number(tokens[i+1]));
                g.vertex_element_index.push(g.vertex_start_index - 1 + Number(tokens[i+4]));
                g.vertex_element_index.push(g.vertex_start_index - 1 + Number(tokens[i+7]));
                
                g.vertex_texture_index.push(g.texturecoord_start_index - 1 + Number(tokens[i+2]));
                g.vertex_texture_index.push(g.texturecoord_start_index - 1 + Number(tokens[i+5]));
                g.vertex_texture_index.push(g.texturecoord_start_index - 1 + Number(tokens[i+8]));
                
                g.vertex_normal_index.push(g.normal_start_index - 1 + Number(tokens[i+3]));
                g.vertex_normal_index.push(g.normal_start_index - 1 + Number(tokens[i+6]));
                g.vertex_normal_index.push(g.normal_start_index - 1 + Number(tokens[i+9]));
                i = i + 10;
            }
            else{
                console.log("Error, unexpected " + tokens[i]);
                i++;
            }
        }
        g.vertex_num_indices = (vbo_vertices.length / 3) - g.vertex_start_index;
        g.normal_num_indices = (vbo_normals.length / 3) - g.normal_start_index;
        g.texturecoord_num_indicies = (vbo_texcoords.length / 2) - g.texturecoord_start_index;
        g.diffuse = color;
        g.ambient = [color[0] * 0.3, color[1] * 0.3, color[2] * 0.3];
        
        g.name = location;
        g.localScale = scale;
        g.translate = translate;
        g.rotation = rotate;
        g.root = scene_graph;
        scene_graph.children.push(g);
        if(useTexture != null){
            init_texture(g, useTexture);
        }
        if(useCubemap != null){
            init_cubeMap(g, useCubemap);
        }
        CreateBuffers();
        draw_scene();
        
        update_heirarchy();
    }
    xhr.send(); 
}
function tokenize(inputString){
    tokens = [];
    currentToken = "";
    for(var spot = 0; spot < inputString.length; spot++){
        var charAt = inputString.charAt(spot);
        if(' \t\n\r\v'.indexOf(charAt) > -1){ 
            if(currentToken != ""){
                tokens.push(currentToken);
                currentToken = "";
            }
        }
        else if(charAt == "/"){
            tokens.push(currentToken);
            currentToken = "";
        }
        else{
            currentToken = currentToken + charAt;
        }
    }
    return tokens;
}
///////////////////////////////////////////////////////////////

/**
* Called from index.html after everything has loaded with
* onload within the body. Initializes everything needed.
*/
function webGLStart() {
    var canvas = document.getElementById("lab5-canvas");
    initGL(canvas);
    initShaders();
    gl.enable(gl.DEPTH_TEST);
    
    shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, "aVertexPosition");
    gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);
    shaderProgram.vertexNormalAttribute = gl.getAttribLocation(shaderProgram, "aVertexNormal");
    gl.enableVertexAttribArray(shaderProgram.vertexNormalAttribute);
    shaderProgram.vertexColorAttribute = gl.getAttribLocation(shaderProgram, "aVertexColor");
    gl.enableVertexAttribArray(shaderProgram.vertexColorAttribute);
    shaderProgram.vertexTextureCoordinateAttribute = gl.getAttribLocation(shaderProgram, "aVertexTexCoord");
    gl.enableVertexAttribArray(shaderProgram.vertexTextureCoordinateAttribute);
    
    shaderProgram.textureUniform = gl.getUniformLocation(shaderProgram, "myTexture");
    shaderProgram.normalUniform = gl.getUniformLocation(shaderProgram, "normalMap");
    shaderProgram.cubeMapTextureUniform = gl.getUniformLocation(shaderProgram, "myCubeMap");
    shaderProgram.use_textureUniform = gl.getUniformLocation(shaderProgram, "use_texture");
    shaderProgram.use_normalMapUniform = gl.getUniformLocation(shaderProgram, "use_normalMap");
    shaderProgram.use_reflectionMappingUniform = gl.getUniformLocation(shaderProgram, "use_reflectionMapping");
    shaderProgram.use_lightingUniform = gl.getUniformLocation(shaderProgram, "use_lighting");

    shaderProgram.mMatrixUniform = gl.getUniformLocation(shaderProgram, "uMMatrix");    
    shaderProgram.vMatrixUniform = gl.getUniformLocation(shaderProgram, "uVMatrix");
    shaderProgram.pMatrixUniform = gl.getUniformLocation(shaderProgram, "uPMatrix");
    shaderProgram.nMatrixUniform = gl.getUniformLocation(shaderProgram, "uNMatrix");
    shaderProgram.v2wMatrixUniform = gl.getUniformLocation(shaderProgram, "v2wMatrix");
    shaderProgram.lightPosition = gl.getUniformLocation(shaderProgram, "light_position");
    shaderProgram.lightAmbient = gl.getUniformLocation(shaderProgram, "light_ambient");
    shaderProgram.lightDiffuse = gl.getUniformLocation(shaderProgram, "light_diffuse");
    shaderProgram.lightSpecular = gl.getUniformLocation(shaderProgram, "light_specular");
    shaderProgram.materialAmbient = gl.getUniformLocation(shaderProgram, "material_ambient");
    shaderProgram.materialDiffuse = gl.getUniformLocation(shaderProgram, "material_diffuse");
    shaderProgram.materialSpecular = gl.getUniformLocation(shaderProgram, "material_specular");
    shaderProgram.materialShininess = gl.getUniformLocation(shaderProgram, "material_shininess");
    shaderProgram.textureOffset = gl.getUniformLocation(shaderProgram, "texture_offset");
    shaderProgram.textureScale = gl.getUniformLocation(shaderProgram, "texture_scale");
    
    
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    
    initScene();
    
    //document.addEventListener('mousedown', onDocumentMouseDown,false);
    document.addEventListener('keydown', onKeyDown, false);
}

/**
* Creates the 2 total buffers used in this implementation.
* 1 for the vertex buffer for the primitive type gl.TRIANGLES, 
* and 1 more for the color buffer
*/
function CreateBuffers() {
    
    VertexPositionBuffer = gl.createBuffer();
    VertexPositionBuffer.numItems = vbo_vertices.length / 3;    
    
    VertexColorBuffer = gl.createBuffer();
    VertexColorBuffer.numItems = vbo_colors.length / 4;    

    VertexNormalBuffer = gl.createBuffer();
    VertexNormalBuffer.numItems = vbo_normals.length / 3;

    VertexTextureCoordsBuffer = gl.createBuffer();
    VertexTextureCoordsBuffer.numitems = vbo_texcoords.length / 2;

    VertexIndexBuffer = gl.createBuffer();
        
    gl.bindBuffer(gl.ARRAY_BUFFER, VertexPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vbo_vertices), gl.STATIC_DRAW);    
    
    gl.bindBuffer(gl.ARRAY_BUFFER, VertexColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vbo_colors), gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, VertexNormalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vbo_normals), gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, VertexTextureCoordsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vbo_texcoords), gl.STATIC_DRAW);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, VertexIndexBuffer);

}


/**
* Initializes the viewport for drawing.
*/
function initScene() {

    //document.getElementById("lab1-canvas").style.width = window.innerWidth;
    //document.getElementById("lab1-canvas").style.height = window.innerHeight;
    gl.canvasWidth = document.getElementById("lab5-canvas").width; //window.innerWidth;
    gl.canvasHeight = document.getElementById("lab5-canvas").height; //window.innerHeight;
    vp_minX = 0; vp_maxX = gl.canvasWidth;  vp_width = vp_maxX- vp_minX+1; 
    vp_minY = 0; vp_maxY = gl.canvasHeight; vp_height = vp_maxY-vp_minY+1; 
    
    gl.viewport(vp_minX, vp_minY, vp_width, vp_height); 
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    
    lightGO = create_sphere(30, 30, [1, 1, 1]);
    lightGO.name = "Light";
    scene_graph.children.push(lightGO);
    lightGO.root = scene_graph;
    lightGO.translate = lightPosition;
    lightGO.localScale = [5, 5,5];
    lightGO.ambient = [1.0, 1.0, 0.0];

    var kb = create_klein_bottle(20, 20, [1, 1, 1]);
    kb.name = "KleinBottle";
    kb.localScale = [20, 20, 20];
    kb.translate = [0, 0, 0];
    scene_graph.children.push(kb);
    kb.root = scene_graph;
    init_texture(kb, "./brick.jpg");
    kb.ambient = [0.1, 0.1, 0.1];
  
    var skybox = create_cube(1, [1, 0, 0]);
    skybox.name = "skybox";
    skybox.localScale = [10000, 10000, 10000];
    skybox.translate = [0, 0, 0];
    scene_graph.children.push(skybox);
    skybox.root = scene_graph;
    init_texture(skybox, "./skybox.jpg");
    skybox.ambient = [1.0, 1.0, 1.0];

    var reflectSkybox = create_sphere(30, 30, [1, 1, 1]);
    reflectSkybox.name = "reflectSkybox";
    scene_graph.children.push(reflectSkybox);
    reflectSkybox.root = scene_graph;
    reflectSkybox.localScale = [15, 15, 15];
    reflectSkybox.ambient = [0.8, 0.8, 0.8];
    init_cubeMap(reflectSkybox, "./skybox");

    var globe = create_sphere(50, 50, [1, 1, 1]);
    globe.name = "globe";
    reflectSkybox.children.push(globe);
    globe.root = reflectSkybox;
    globe.localScale = [.75, .75, .75];
    globe.translate = [-3, 0, 0];
    globe.ambient = [0.8, 0.8, 0.8];
    init_texture(globe, "./4096_earth.jpg");

    var moon = create_sphere(50, 50, [1, 1, 1]);
    moon.name = "moon";
    globe.children.push(moon);
    moon.root = globe;
    moon.localScale = [.5, .5, .5];
    moon.translate = [1.75, 0, 0];
    moon.ambient = [0.8, 0.8, 0.8];
    init_texture(moon, "./8k_moon.jpg");

    loadOBJfile("./rock.obj", [35, -10, 0], [0, 0, 0], [3, 3, 3], [1, 1, 1], null, "./skybox");
    //loadOBJfile("./TheCity.obj", [0, -50, 200], [0, 0, 0], [0.1, 0.1, 0.1], [.2, .4, 0.8]);
    
    // Create the three axes for reference
    var cube = create_cube(1, [1, 0, 0]);
    cube.name = "X-axis";
    cube.localScale = [40, .5, .5];
    cube.translate = [10, 0, 0];
    scene_graph.children.push(cube);
    cube.root = scene_graph;
    
    cube = create_cube(1, [0, 1, 0]);
    cube.name = "Y-axis";
    cube.localScale = [.5, 40, .5];
    cube.translate = [0, 10, 0];
    scene_graph.children.push(cube);
    cube.root = scene_graph;

    cube = create_cube(1, [0, 0, 1]);
    cube.name = "Z-axis";
    cube.localScale = [.5, .5, 40];
    cube.translate = [0, 0, 10];
    scene_graph.children.push(cube);
    cube.root = scene_graph;

    var goodBrick = create_cube(1, [1, 1, 1]);
    goodBrick.name = "goodBrick";
    goodBrick.localScale = [50, 50, 50];
    goodBrick.translate = [20, 30, 50];
    scene_graph.children.push(goodBrick);
    goodBrick.root = scene_graph;
    goodBrick.textureScale = [5, 5];
    
    init_texture(goodBrick, "./goodBrick.jpg");
    init_normalMap(goodBrick, "./goodBrickNormal.jpg");

 
/*
    // Create all the parts of our robot dude
    base = create_cube(1, [0.3, 0.3, 0.3]);
    base.name = "Base";
    scene_graph.children.push(base);
    base.root = scene_graph;
    base.localScale = [20, 7, 7];
    base.rotation = [0, 0, 0];
    base.translate = [-20, -46, 43];

    chest = create_cube(1, [1, .8, 0]);
    chest.name = "Chest";
    base.children.push(chest);
    chest.root = base;
    chest.localScale = [8 * (1/chest.root.global_scale()[0]), 20 * (1/chest.root.global_scale()[1]), 5 * (1/chest.root.global_scale()[2])];
    chest.translate = [0*(1/chest.root.global_scale()[0]), 10*(1/chest.root.global_scale()[1]), 0* (1/chest.root.global_scale()[2])];

    head = create_sphere(5, 10, 10, [.5, 0, .5]);
    head.name = "Head";
    chest.children.push(head);
    head.root = chest;
    head.localScale = [1.4 * (1/head.root.global_scale()[0]), 1 * (1/head.root.global_scale()[1]), 1.4 * (1/head.root.global_scale()[2])];
    head.translate = [0 * (1/head.root.global_scale()[0]), 15 * (1/head.root.global_scale()[1]), 0 * (1/head.root.global_scale()[2])];

    rightArm = create_cylinder(2, 3, 8, 10, 2, [.1, .25, .6]);
    rightArm.name = "Right arm";
    chest.children.push(rightArm);
    rightArm.root = chest;
    rightArm.localScale = [(1/rightArm.root.global_scale()[0]), (1/rightArm.root.global_scale()[1]), (1/rightArm.root.global_scale()[2])];
    rightArm.translate = [-7*(1/rightArm.root.global_scale()[0]), 5*(1/rightArm.root.global_scale()[1]), 0*(1/rightArm.root.global_scale()[2])];
    rightArm.rotation = [0, 0, -45];

    rightForearm = create_cylinder(2.5, 2, 6, 10, 2, [0, 1, 0.5]);
    rightForearm.name = "Right forearm";
    rightArm.children.push(rightForearm);
    rightForearm.root = rightArm;
    rightForearm.localScale = [(1/rightForearm.root.global_scale()[0]), 1*(1/rightForearm.root.global_scale()[1]), (1/rightForearm.root.global_scale()[2])];
    rightForearm.translate = [1*(1/rightForearm.root.global_scale()[0]), -6*(1/rightForearm.root.global_scale()[1]), 0*(1/rightForearm.root.global_scale()[2])];
    rightForearm.rotation = [0, 0, 20];
    

    rightHand = create_sphere(3, 20, 20, [0.6, 0.1, 0.1]);
    rightHand.name = "Right hand";
    rightForearm.children.push(rightHand);
    rightHand.root = rightForearm;
    rightHand.localScale = [(1/rightHand.root.global_scale()[0]), (1/rightHand.root.global_scale()[1]), (1/rightHand.root.global_scale()[2])];
    rightHand.translate = [0.5*(1/rightHand.root.global_scale()[0]), -5*(1/rightHand.root.global_scale()[1]), 0*(1/rightHand.root.global_scale()[2])];

    leftArm = create_cylinder(2, 3, 8, 10, 2, [.1, .25, .6]);
    leftArm.name = "Left arm";
    chest.children.push(leftArm);
    leftArm.root = chest;
    leftArm.localScale = [(1/leftArm.root.global_scale()[0]), (1/leftArm.root.global_scale()[1]), (1/leftArm.root.global_scale()[2])];
    leftArm.translate = [7*(1/leftArm.root.global_scale()[0]), 7*(1/leftArm.root.global_scale()[1]), 0*(1/leftArm.root.global_scale()[2])];
    leftArm.rotation = [0, 0, 100];

    leftForearm = create_cylinder(2.5, 2, 6, 10, 2, [0, 1, 0.5]);
    leftForearm.name = "Left forearm";
    leftArm.children.push(leftForearm);
    leftForearm.root = leftArm;
    leftForearm.localScale = [(1/leftForearm.root.global_scale()[0]), (1/leftForearm.root.global_scale()[1]), (1/leftForearm.root.global_scale()[2])];
    leftForearm.translate = [2*(1/leftForearm.root.global_scale()[0]), -6*(1/leftForearm.root.global_scale()[1]), 0*(1/leftForearm.root.global_scale()[2])];
    leftForearm.rotation = [0, 0, 40];

    leftHand = create_sphere(3, 20, 20, [0.6, 0.1, 0.1]);
    leftHand.name = "Left hand";
    leftForearm.children.push(leftHand);
    leftHand.root = leftForearm;
    leftHand.localScale = [(1/leftHand.root.global_scale()[0]), (1/leftHand.root.global_scale()[1]), (1/leftHand.root.global_scale()[2])];
    leftHand.translate = [0.5*(1/leftHand.root.global_scale()[0]), -5*(1/leftHand.root.global_scale()[1]), 0*(1/leftHand.root.global_scale()[2])];
*/
    CreateBuffers();
    draw_scene();
    update_heirarchy();
    setInterval(function(){ update() }, 32);
}

function update(){
    var globe = findGameobject("globe");
    var reflectSkybox = findGameobject("reflectSkybox");
    globe.rotation = [globe.rotation[0], globe.rotation[1]+5, globe.rotation[2]];
    reflectSkybox.rotation = [reflectSkybox.rotation[0], reflectSkybox.rotation[1]+1, reflectSkybox.rotation[2]];
    CreateBuffers();
    draw_scene();
}
/**
* Draws the scene using draw_scene_elements after preparing the viewport and clearing the buffers.
*/
function draw_scene() {
    vp_minX = 0; vp_maxX = gl.canvasWidth;  vp_width = vp_maxX- vp_minX+1; 
    vp_minY = 0; vp_maxY = gl.canvasHeight; vp_height = vp_maxY-vp_minY+1; 
    gl.viewport(vp_minX, vp_minY, vp_width, vp_height); 
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    vMatrix = mat4.identity(vMatrix);    
    v2wMatrix = mat4.identity(v2wMatrix);
    pMatrix = mat4.identity(pMatrix);
    
    pMatrix = mat4.perspective(60, 1.0, 0.1, 100000, pMatrix);  // set up the projection matrix 
    // Find the rotation matrix with the current roll pitch yaw, then multiply by lookat
    var rotM;
    rotM = mat4.create();
    rotM = mat4.identity(rotM);    
    rotM = mat4.rotate(rotM, degToRad(cam_pitch_yaw_roll[0]), [1, 0, 0], rotM);
    rotM = mat4.rotate(rotM, degToRad(cam_pitch_yaw_roll[1]), [0, 1, 0], rotM);
    rotM = mat4.rotate(rotM, degToRad(cam_pitch_yaw_roll[2]), [0, 0, 1], rotM);
    vMatrix = mat4.lookAt(cam_pos, cam_look_at, [0,1,0], vMatrix);	// set up the view matrix
    vMatrix = mat4.multiply(rotM, vMatrix, vMatrix);
    v2wMatrix = mat4.multiply(v2wMatrix, vMatrix);
    v2wMatrix = mat4.inverse(v2wMatrix);
    gl.uniformMatrix4fv(shaderProgram.vMatrixUniform, false, vMatrix);
    gl.uniformMatrix4fv(shaderProgram.pMatrixUniform, false, pMatrix);
    gl.uniformMatrix4fv(shaderProgram.v2wMatrixUniform, false, v2wMatrix);
    
    gl.uniform4f(shaderProgram.lightPosition, lightPosition[0], lightPosition[1], lightPosition[2], 1.0);
    gl.uniform3f(shaderProgram.lightAmbient, lightAmbient[0], lightAmbient[1], lightAmbient[2]);
    gl.uniform3f(shaderProgram.lightDiffuse, lightDiffuse[0], lightDiffuse[1], lightDiffuse[2]);
    gl.uniform3f(shaderProgram.lightSpecular, lightSpecular[0], lightSpecular[1], lightSpecular[2]);

    var current_matrix = mat4.create();
    current_matrix = mat4.identity(current_matrix);

    draw_scene_elements(scene_graph, current_matrix);
}
/**
*   Draws the scene recursively, passing the matrix from all parent nodes above it as the "world matrix"
*   to be multiplied. Draw the element with that world element, then draw all children from the gameObject
*   with the new relative world matrix.
*/
function draw_scene_elements(gameObject, relative_world_matrix){   
    gameObject.draw(relative_world_matrix);
    var new_relative_world_matrix = mat4.create();
    new_relative_world_matrix = mat4.multiply(relative_world_matrix, gameObject.get_m_matrix(), new_relative_world_matrix);
    for(var i = 0; i < gameObject.children.length; i++){
        if(gameObject.children[i].enabled){
            draw_scene_elements(gameObject.children[i], new_relative_world_matrix);
        }
    }
}

/**
 * Updates the visible heirarchy in the div id=heirarchy
 * 
 */
function update_heirarchy(){
    var div = document.getElementById("heirarchy");
    var str = "";
   
    str = update_heirarchy_recursive(scene_graph, str);
    div.innerHTML = str;

    
    var toggler = document.getElementsByClassName("caret");
    var i;
    for (i = 0; i < toggler.length; i++) {
        toggler[i].parentElement.querySelector(".nested").classList.toggle("active");
        toggler[i].classList.toggle("caret-down");
        toggler[i].addEventListener("click", function() {
            //this.parentElement.querySelector(".nested").classList.toggle("active");
            //this.classList.toggle("caret-down");
        });
    }
    toggler = document.getElementsByName("tree");
    i;
    for (i = 0; i < toggler.length; i++) {
        toggler[i].addEventListener("click", function() {
            selectObject(this);
        });
    }
    
}
function transformGO(){
    if(selected_gameObject != null){
        selected_gameObject.translate[0] = Number(document.getElementById("xPos").value);
        selected_gameObject.translate[1] = Number(document.getElementById("yPos").value);
        selected_gameObject.translate[2] = Number(document.getElementById("zPos").value);

        selected_gameObject.rotation[0] = Number(document.getElementById("xRot").value);
        selected_gameObject.rotation[1] = Number(document.getElementById("yRot").value);
        selected_gameObject.rotation[2] = Number(document.getElementById("zRot").value);

        selected_gameObject.localScale[0] = Number(document.getElementById("xScale").value);
        selected_gameObject.localScale[1] = Number(document.getElementById("yScale").value);
        selected_gameObject.localScale[2] = Number(document.getElementById("zScale").value);

        selected_gameObject.textureOffset[0] = Number(document.getElementById("texOffsetX").value);
        selected_gameObject.textureOffset[1] = Number(document.getElementById("texOffsetY").value);
        selected_gameObject.textureScale[0] = Number(document.getElementById("texScaleX").value);
        selected_gameObject.textureScale[1] = Number(document.getElementById("texScaleY").value);
        CreateBuffers();
        draw_scene();
    }
}
function selectObject(object){
    var go = findGameobject(object.innerHTML);
    selected_gameObject = go;
    document.getElementById("xPos").value = go.translate[0];
    document.getElementById("yPos").value = go.translate[1];
    document.getElementById("zPos").value = go.translate[2];
    
    document.getElementById("xRot").value = go.rotation[0];
    document.getElementById("yRot").value = go.rotation[1];
    document.getElementById("zRot").value = go.rotation[2];
    
    document.getElementById("xScale").value = go.localScale[0];
    document.getElementById("yScale").value = go.localScale[1];
    document.getElementById("zScale").value = go.localScale[2];

    document.getElementById("texOffsetX").value = go.textureOffset[0];
    document.getElementById("texOffsetY").value = go.textureOffset[1];
    document.getElementById("texScaleX").value = go.textureScale[0];
    document.getElementById("texScaleY").value = go.textureScale[1];
}
function findGameobject(name){
    var theObject = null;
    var objectsToCheck = [scene_graph];
    while(objectsToCheck.length > 0 && theObject == null){
        var front = objectsToCheck.pop();
        if(front.name == name){
            theObject = front;
        }
        else{
            for(var i = 0; i < front.children.length; i++){
                objectsToCheck.push(front.children[i]);
            }
        }
    }

    return theObject;
}
function update_heirarchy_recursive(startingObject, str){
    if(startingObject.children.length > 0){
        str = str + "<li><span name='tree' class='caret' id='"+startingObject.name+"'>"+startingObject.name+"</span><ul class='nested'>";
        for(var i = 0; i < startingObject.children.length; i++){
            str = update_heirarchy_recursive(startingObject.children[i], str);
        }
        str = str + "</ul></li>";
    }
    else{
        str = str + "<li name='tree' id='"+startingObject.name+"'>"+startingObject.name+"</li>";
    }
    return str;
}


/**
* Captures all mouse down events. This is when
* something should be drawn, so we need to add
* vertices and colors to the correct buffer based
* on the current polygon and color modes.
*/
function onDocumentMouseDown( event ) {
    event.preventDefault();
    document.addEventListener( 'mousemove', onDocumentMouseMove, false );
    document.addEventListener( 'mouseup', onDocumentMouseUp, false );
    document.addEventListener( 'mouseout', onDocumentMouseOut, false );
	var p = document.getElementById("lab5-body");
    var leftMargin = getStyle(p, 'margin-left').substring(0, getStyle(p, 'margin-left').length - 2);
    var topMargin = getStyle(p, 'margin-top').substring(0, getStyle(p, 'margin-top').length - 2);
    
	var NDC_X = (event.clientX - leftMargin - vp_minX)/vp_width*2 - 1; 
	var NDC_Y = -((event.clientY - topMargin - vp_minY)/vp_height*2 - 1);   
    
 }

 function renderPoints(){
    if(selected_gameObject != null){
        selected_gameObject.drawType = "POINTS";
        CreateBuffers();
        draw_scene();
    }
 }
 function renderLines(){
    if(selected_gameObject != null){
        selected_gameObject.drawType = "LINES";
        CreateBuffers();
        draw_scene();
    }
 }
 function renderTriangles(){
    if(selected_gameObject != null){
        selected_gameObject.drawType = "TRIANGLES";
        CreateBuffers();
        draw_scene();
    }
 }
 function toggleLighting(){
    if(selected_gameObject != null){
        selected_gameObject.useLighting = !selected_gameObject.useLighting;
        CreateBuffers();
        draw_scene();
    }
 }
 function toggleTexture(){
    if(selected_gameObject != null && selected_gameObject.texture != null){
        selected_gameObject.useTexture = !selected_gameObject.useTexture;
        CreateBuffers();
        draw_scene();
    }
 }
 function toggleNormalMap(){
    if(selected_gameObject != null && selected_gameObject.normalMap != null){
        selected_gameObject.useNormalMap = !selected_gameObject.useNormalMap;
        CreateBuffers();
        draw_scene();
    }
 }
 function toggleCubemap(){
    if(selected_gameObject != null && selected_gameObject.cubeMap != null){
        selected_gameObject.useReflectionMapping = !selected_gameObject.useReflectionMapping;
        CreateBuffers();
        draw_scene();
    }
 }

/** 
* Handles mouse move events. Rotates the selected object if mouse_down=true
*/
function onDocumentMouseMove( event ) {
    var mouseX = event.clientX;
    var mouseY = event.clientY;
    var p = document.getElementById("lab5-body");
    var leftMargin = getStyle(p, 'margin-left').substring(0, getStyle(p, 'margin-left').length - 2);
    var topMargin = getStyle(p, 'margin-top').substring(0, getStyle(p, 'margin-top').length - 2);
    
	var NDC_X = (event.clientX - leftMargin - vp_minX)/vp_width*2 - 1; 
	var NDC_Y = -((event.clientY - topMargin - vp_minY)/vp_height*2 - 1);
    
}

/** 
* Handles mouse up events. Not used in this project.
*/
function onDocumentMouseUp( event ) {
    document.removeEventListener( 'mousemove', onDocumentMouseMove, false );
    document.removeEventListener( 'mouseup', onDocumentMouseUp, false );
    document.removeEventListener( 'mouseout', onDocumentMouseOut, false );
    mouse_down = false;
}

/**
* Handles mouse out events. Not used in this project.
*/
function onDocumentMouseOut( event ) {
    document.removeEventListener( 'mousemove', onDocumentMouseMove, false );
    document.removeEventListener( 'mouseup', onDocumentMouseUp, false );
    document.removeEventListener( 'mouseout', onDocumentMouseOut, false );
}

/**
* Handles all keystroke events on key down. Used to
* determine if we need to change color/polygon mode
* or if the screen should be redrawn or cleared.
*/
function onKeyDown(event) {
    
    // Calculate world right, up, forward for camera
    var dirs = get_eye_vectors();
    switch(event.keyCode)  {
        case 33:
            // page up
            cam_pos = vec3.add(cam_pos, dirs[1], cam_pos);
            cam_look_at = vec3.add(cam_look_at, dirs[1], cam_look_at);
            CreateBuffers();
            draw_scene();
            break;
        case 34: 
            // page down
            cam_pos = vec3.subtract(cam_pos, dirs[1], cam_pos);
            cam_look_at = vec3.subtract(cam_look_at, dirs[1], cam_look_at);
            CreateBuffers();
            draw_scene();
            break;
        case 37:
            // left arrow
            cam_pos = vec3.subtract(cam_pos, dirs[0], cam_pos);
            cam_look_at = vec3.subtract(cam_look_at, dirs[0], cam_look_at);
            CreateBuffers();
            draw_scene();
            break;
        case 38:
            // up arrow
            cam_pos = vec3.add(cam_pos, dirs[2], cam_pos);
            cam_look_at = vec3.add(cam_look_at, dirs[2], cam_look_at);
            CreateBuffers();
            draw_scene();
            break;
        case 39:
            // right arrow
            cam_pos = vec3.add(cam_pos, dirs[0], cam_pos);
            cam_look_at = vec3.add(cam_look_at, dirs[0], cam_look_at);
            CreateBuffers();
            draw_scene();
            break;
        case 40:
            // down arrow
            cam_pos = vec3.subtract(cam_pos, dirs[2], cam_pos);
            cam_look_at = vec3.subtract(cam_look_at, dirs[2], cam_look_at);
            CreateBuffers();
            draw_scene();
            break;
        case 82:
            // R
            if (event.shiftKey) {
                cam_pitch_yaw_roll = vec3.add(cam_pitch_yaw_roll, [0, 0, 3], cam_pitch_yaw_roll);
            }
            else{
                cam_pitch_yaw_roll = vec3.add(cam_pitch_yaw_roll, [0, 0, -3], cam_pitch_yaw_roll);
            }
            CreateBuffers();
            draw_scene();
            break;
        case 80:
            // P
            if (event.shiftKey) {
                cam_pitch_yaw_roll = vec3.add(cam_pitch_yaw_roll, [3, 0, 0], cam_pitch_yaw_roll);
            }
            else{
                cam_pitch_yaw_roll = vec3.add(cam_pitch_yaw_roll, [-3, 0, 0], cam_pitch_yaw_roll);
            }
            CreateBuffers();
            draw_scene();
            break;
        case 89:
            // Y
            if (event.shiftKey) {
                cam_pitch_yaw_roll = vec3.add(cam_pitch_yaw_roll, [0, 3, 0], cam_pitch_yaw_roll);
            }
            else{
                cam_pitch_yaw_roll = vec3.add(cam_pitch_yaw_roll, [0, -3, 0], cam_pitch_yaw_roll);
            }
            CreateBuffers();
            draw_scene();
            break;
        case 83:
            // S
            lightPosition = [
                lightPosition[0], 
                lightPosition[1] - 0.5, 
                lightPosition[2]];
            lightGO.translate = lightPosition;
            CreateBuffers();
            draw_scene(); 
            break;
        case 87:
            // W
            lightPosition = [
                lightPosition[0], 
                lightPosition[1] + 0.5, 
                lightPosition[2]];
                lightGO.translate = lightPosition;
            CreateBuffers();
            draw_scene();              
            break;
        case 68:
            // D
            lightPosition = [
                lightPosition[0] - 0.5, 
                lightPosition[1], 
                lightPosition[2]];
                lightGO.translate = lightPosition;
            CreateBuffers();
            draw_scene(); 
            break; 
        case 65:
            // A
            lightPosition = [
                lightPosition[0] + 0.5, 
                lightPosition[1], 
                lightPosition[2]];
                lightGO.translate = lightPosition;
            CreateBuffers();
            draw_scene(); 
            break;      
        case 69:
            // E
            lightPosition = [
                lightPosition[0], 
                lightPosition[1], 
                lightPosition[2] + 0.5];
                lightGO.translate = lightPosition;
            CreateBuffers();
            draw_scene(); 
            break; 
        case 81:
            // Q
            lightPosition = [
                lightPosition[0], 
                lightPosition[1], 
                lightPosition[2] - 0.5];
                lightGO.translate = lightPosition;
            CreateBuffers();
            draw_scene(); 
            break;          
    }
}

/**
 * Uses cam_pos and cam_look_at to determine current
 * forward, right, and up vectors  
 */
function get_eye_vectors(){
    var rot = mat4.create();
    rot = mat4.identity(rot);
    rot = mat4.rotate(rot, degToRad(cam_pitch_yaw_roll[0]), [1, 0, 0], rot);
    rot = mat4.rotate(rot, degToRad(cam_pitch_yaw_roll[1]), [0, 1, 0], rot);
    rot = mat4.rotate(rot, degToRad(cam_pitch_yaw_roll[2]), [0, 0, 1], rot);

    var m = mat4.create();
    m = mat4.identity(m);
    m = mat4.lookAt(cam_pos, cam_look_at, [0,1,0], m);
    m = mat4.toRotationMat(m, m);
    m = mat4.multiply(rot, m, m);
    // Fix for the forward and right z being backwards
    m[2] = -m[2];
    m[10] = -m[10];
    
    var right, up, forward;
    right = vec3.create();
    up = vec3.create();
    forward = vec3.create();
    mat4.multiplyVec3(m, [1, 0, 0], right);
    mat4.multiplyVec3(m, [0, 1, 0], up);
    mat4.multiplyVec3(m, [0, 0, 1], forward);
    return [right, up, forward];
}

var old_color = [];
/** 
 *  Highlights a gameObject white so the user knows which is selected.
 */
function highlight(go){
    if(go == selected_gameObject) return;
    if(selected_gameObject != null){
        selected_gameObject.set_color(old_color);
    }
    if(go != null){
        old_color = go.get_color();
        go.set_color([1, 1, 1]);
    }
    selected_gameObject = go;
    CreateBuffers();
    draw_scene(); 
}

function selectNone(){
    highlight(null);
}

/**
 * Resets positions of all objects in scene.
 */
function reset(){
    base.translate = [0, -20, 0];
    chest.translate = [0*(1/chest.root.global_scale()[0]), 10*(1/chest.root.global_scale()[1]), 0* (1/chest.root.global_scale()[2])];
    head.translate = [0 * (1/head.root.global_scale()[0]), 15 * (1/head.root.global_scale()[1]), 0 * (1/head.root.global_scale()[2])];
    rightArm.translate = [-7*(1/rightArm.root.global_scale()[0]), 5*(1/rightArm.root.global_scale()[1]), 0*(1/rightArm.root.global_scale()[2])];
    rightForearm.translate = [1*(1/rightForearm.root.global_scale()[0]), -6*(1/rightForearm.root.global_scale()[1]), 0*(1/rightForearm.root.global_scale()[2])];
    rightHand.translate = [0.5*(1/rightHand.root.global_scale()[0]), -5*(1/rightHand.root.global_scale()[1]), 0*(1/rightHand.root.global_scale()[2])];
    leftArm.translate = [7*(1/leftArm.root.global_scale()[0]), 7*(1/leftArm.root.global_scale()[1]), 0*(1/leftArm.root.global_scale()[2])];
    leftForearm.translate = [2*(1/leftForearm.root.global_scale()[0]), -6*(1/leftForearm.root.global_scale()[1]), 0*(1/leftForearm.root.global_scale()[2])];
    leftHand.translate = [0.5*(1/leftHand.root.global_scale()[0]), -5*(1/leftHand.root.global_scale()[1]), 0*(1/leftHand.root.global_scale()[2])];
    selectNone();
}