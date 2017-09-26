import tensorflow as tf
import os, argparse
import numpy as np

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--frozen_model_filename", default="frozen_model.pb", type=str, help="Frozen model file to import")
    #parser.add_argument("--image",help="Test Image")
    #FLAGS,unparsed = parser.parse_known_args()

    # We use our "load_graph" function
    #graph = load_graph(FLAGS.frozen_model_filename)
    graph = load_graph('../chpk/frozen_model.pb')
    sess = tf.Session()
    op = sess.graph.get_operations() 
    arr = []
    for m in op:
        arr.append(m.values())
    # We can verify that we can access the list of operations in the graph
    #for op in graph.get_operations():
        #print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
        
    #We access the input and output nodes 
    #x = graph.get_tensor_by_name('batch_processing/decode_jpeg_3/DecodeJpeg:0') # Input tensor
    #y = graph.get_tensor_by_name('prefix/softmax:0')
    
    #We launch a Session
    #with tf.Session(graph=graph) as sess:
        #Note: we didn't initialize/restore anything, everything is stored in the graph_def
    #    image_data = tf.gfile.FastGFile('/Users/gaurav.kaila/Documents/Projects/Image_Recognition_App/data_dir/extras/Early_Blight/IMG_20170912_142506463.jpg','rb').read()
    #    prediction = sess.run(graph.get_tensor_by_name('prefix/softmax:0'),feed_dict={x:image_data})
    #    predictions = np.squeeze(prediction)
    #    print (predictions)
