Author: Skylar Wurster
Project 4
Professor Shen
Realtime Graphics

==================================================================
INSTRUCTIONS:

This application renders 3D objects with a single light in an OpenGL scene. 
The light position can be moved around the scene with QWEASD, and lighting
is calculated using the Phong shading model, though Gouraud has been implemented
as well, and can be seen by copying the shaders in GouraudShading.txt to the
shaders in the index.html file.
Further, you may rotate the camera using y/Y, r/R, p/P, or move the camera with 
arrow keys and pageup/pagedown.

Press the following keyboard keys for these effects:

y/Y - yaw the camera +/- 3 degrees
p/P - pitch the camera +/- 3 degrees
r/R - roll the camera +/- 3 degrees

Up arrow - move the camera forward
Down arrow - move the camera backward
Right arrow - move the camera right
Left arrow - move the camera left
Page up - move the camera up
Page down - move the camera down

W - move the light up
S - move the light down
A - move the light left
D - move the light right
Q - move the light backward
E - move the light forward


SPECIAL NOTE: Due to CORS issues with loading local files, a 
workaround is required!!! I recommend creating a local server
in this folder in order to bypass the issue, then navigate to 
127.0.0.1 to load index.html. A server can be created using a 
google chrome plugin
https://chrome.google.com/webstore/detail/web-server-for-chrome/ofhbbkphhbklhfoeikjpcbhemlocgigb?hl=en
or through python in the command line from this directory: 
python -m http.server 8887       # python 3
python -m SimpleHTTPServer       # python 2
according to the class website.
Then loading the files shouldn't be an issue.


==================================================================
IMPLEMENTATION

Three buffers are used - one for the vertices for triangles,
one for the element array for vertices creating triangles,
one for the vertex normals. Though a vbo is created for
color, it is not used anymore.

Phong shading is implemented using a number of uniform and 
varying variables passed to the shaders. Since Phong uses
per-fragment shading, we have to use varying for values
that need to be averaged between the vertices the fragment
concerns. Any calculation that isn't needed per vertex is 
done in the vertex shader, to be more efficient.

I could not find JSONs for loading, so I wrote a parser for
loading basic OBJ files that use triangular faces, and removed lines
that deal with material properties and defining surfaces. Code for 
this is located in LoadOBJfile(), called from initScene. Models
were downloaded royalty free from gctrader.com. 
