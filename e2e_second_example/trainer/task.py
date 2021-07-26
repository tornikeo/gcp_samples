import tensorflow as tf

def get_data():
    x = tf.random.normal((100,10))
    y = tf.reduce_mean(x, axis=1)
    return x,y

def main():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])
    x, y = get_data()
    model.compile(loss='mse')
    model.fit(x, y)
    
    @tf.function(
        input_signature=[
            tf.TensorSpec((None, 10), tf.float32),
            tf.TensorSpec((None,),tf.string),
        ]
    )
    def serve(row, key):
        tf.print(row)
        output = model(row)
        return {"output":output, 
                'key':key}
    
    tf.saved_model.save(model, 
                        'output', 
                        signatures=serve)
    model.save('output_keras', 
               save_format='tf',
               signatures=serve)
    
if __name__ == '__main__':
    main()
