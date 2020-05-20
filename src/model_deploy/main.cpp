#include "DA7212.h"
#include "accelerometer_handler.h"
#include "mbed.h"
#include "config.h"

#include "magic_wand_model_data.h"


#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"



#include <cmath>
#include <string>
#include<bits/stdc++.h>
#include "uLCD_4DGL.h"

DA7212 audio;
uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;

Thread DNNthread(osPriorityNormal,100*1024); 
int16_t waveform[kAudioTxBufferSize];
int start=0;
EventQueue queue(32 * EVENTS_EVENT_SIZE);
InterruptIn button1(SW2);
InterruptIn button2(SW3);
Thread t;
int pause=1;
int mode=0;      ///0 play music     1  select     2   game
int music=0;     ///0 mary   1 tiger    2  python
int current_mode=0;
int music_now=0;
int si=494;
int la=440;
int sol=392;
int dor=261;
int re=293;
int mi=330;
int fa=349;
int so=392;
int Do=523;

int Mary[25]={si,la,sol,la,si,si,si,la,la,la,si,si,si,si,la,sol,la,si,si,si,la,la,si,la,sol};
int length_Mary[25]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2};
int tiger[32]={dor,re,mi,dor,dor,re,mi,dor,mi,fa,so,mi,fa,so,so,la,so,fa,mi,dor,so,la,so,fa,mi,dor,mi,196,dor,mi,196,dor};
int length_tiger[32]={1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2};
// Return the result of the last prediction
int gesture_index=-1;
int DNN_count;
int cursor_move=-1;
void playNote(int freq)

{

  for(int i = 0; i < kAudioTxBufferSize; i++)

  {

    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));

  }

  audio.spk.play(waveform, kAudioTxBufferSize);

}

/*****************************************music*/
int PredictGesture(float* output) {

  // How many times the most recent gesture has been matched in a row

  static int continuous_count = 0;

  // The result of the last prediction

  static int last_predict = -1;


  // Find whichever output has a probability > 0.8 (they sum to 1)

  int this_predict = -1;

  for (int i = 0; i < label_num; i++) {

    if (output[i] > 0.8) this_predict = i;

  }


  // No gesture was detected above the threshold

  if (this_predict == -1) {

    continuous_count = 0;

    last_predict = label_num;

    return label_num;

  }


  if (last_predict == this_predict) {

    continuous_count += 1;

  } else {

    continuous_count = 0;

  }

  last_predict = this_predict;


  // If we haven't yet had enough consecutive matches for this gesture,

  // report a negative result

  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {

    return label_num;

  }

  // Otherwise, we've seen a positive result, so clear all our variables

  // and report it

  continuous_count = 0;

  last_predict = -1;


  return this_predict;

}

void DNNgesture(){
        //selecting mode:3 option, 1.2 have the same state;
        // Create an area of memory to use for input, output, and intermediate arrays.
        // The size of this will depend on the model you're using, and may need to be
        // determined by experimentation.
        constexpr int kTensorArenaSize = 60 * 1024;
        uint8_t tensor_arena[kTensorArenaSize];
        // Whether we should clear the buffer next time we fetch data
        bool should_clear_buffer = false;
        bool got_data = false;
        // The gesture index of the prediction


        // Set up logging.
        static tflite::MicroErrorReporter micro_error_reporter;

        tflite::ErrorReporter* error_reporter = &micro_error_reporter;


        // Map the model into a usable data structure. This doesn't involve any

        // copying or parsing, it's a very lightweight operation.

        const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
        /*
        if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
        }*/
        // Pull in only the operation implementations we need.
        // This relies on a complete list of all the ops needed by this graph.
        // An easier approach is to just use the AllOpsResolver, but this will
        // incur some penalty in code space for op implementations that are not
        // needed by this graph.

        static tflite::MicroOpResolver<6> micro_op_resolver;
        micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
        tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
        micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(),1); //add missing op

        // Build an interpreter to run the model with

        static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
        tflite::MicroInterpreter* interpreter = &static_interpreter;


        // Allocate memory from the tensor_arena for the model's tensors
        interpreter->AllocateTensors();
        // Obtain pointer to the model's input tensor
        TfLiteTensor* model_input = interpreter->input(0);
        /*
        if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != config.seq_length) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32)) {
        error_reporter->Report("Bad input tensor parameters in model");
        return -1;
        }*/


        int input_length = model_input->bytes / sizeof(float);


        TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
        /*
        if (setup_status != kTfLiteOk) {
            error_reporter->Report("Set up failed\n");
            return -1;
        }
        error_reporter->Report("Set up successful...\n");*/
        while (1) {
            // Attempt to read new data from the accelerometer
            got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                             input_length, should_clear_buffer);
            // If there was no new data,
            // don't try to clear the buffer again and wait until next time
            if (!got_data) {
            should_clear_buffer = false;
            continue;
            }
            // Run inference, and report any error
            TfLiteStatus invoke_status = interpreter->Invoke();
            /*
            if (invoke_status != kTfLiteOk) {
                error_reporter->Report("Invoke failed on index: %d\n", begin_index);
                continue;
            }*/

            // Analyze the results to obtain a prediction
            gesture_index = PredictGesture(interpreter->output(0)->data.f);
            // Clear the buffer next time we read data

            should_clear_buffer = gesture_index < label_num;


            // Produce an output
            /*if (gesture_index < label_num) {
                cursor_move = gesture_index;
            }*/
        }

}
int co=0;
int main() {
  DNNthread.start(&DNNgesture);
  while (1) {
    uLCD.cls();
    DNN_count=50; //BJ:run loop 50 times to get cursor_move 
    cursor_move=-1;
    
    while (DNN_count--) {
      cursor_move = gesture_index;
    }  
      if(cursor_move == 2){
       
        if(mode == 2){
          mode = 0;
        }
        else{
        mode++;
        }    
        co++;
        break;
      }
      else if(cursor_move == 3){
       
        if(mode <= 0){
          mode = 2;
        }
        else{
          mode--;
        }  
        co++;
        break;
      }
      //wait(0.0001);
    uLCD.printf("\n count %d \n",co); 
    uLCD.printf("\n DNN %d \n",cursor_move);  
    uLCD.printf("\n static mode %d \n",mode);
    /********mode1*/
   

    if(button2==0)
      current_mode=mode;
    /******************************mode0*/
    if(current_mode==0){  
      start=1;
      mode=0;
      uLCD.printf("\n mode:music time \n");
      int song[50];
      int noteLength[50];
      int num;
      if(music_now==1){
        for(int i=0;i<25;i++){
          song[i]=Mary[i];
          noteLength[i]=length_Mary[i];
        }
        num=25;
      }
      else
      {
        for(int i=0;i<32;i++){
          song[i]=tiger[i];
          noteLength[i]=length_tiger[i];
        }    
        num=32;    
      }
      
      while(start==1){
       // t.start(callback(&queue, &EventQueue::dispatch_forever));
        if(start==0){     
         uLCD.printf("\n mode selection \n");
         break;
        }
        else{
          for(int i = 0; i < num; i++)
          {
            if(button1==0){
              break;
              start=0;
            }
            else{
              int length = noteLength[i];
              while(length--)
              {
                playNote(song[i]);
                if(length <= 1) wait(1.0);
              }
            }

          }
          start=0;
        }  
      }

    }


  
  else if(current_mode==1){
    mode=1;
    uLCD.printf("\n mode:select music \n") ;     
  }
  else if(current_mode==2){ 
    mode=2;
    uLCD.printf("\n mode:Taiko game \n");
  }
  else{ 
    mode=mode;
    uLCD.printf("\n error! you suck. \n");
  }    
    /*******************************mode1***/
    if(current_mode==1){
      if(gesture_index==0){
        if(music==0)
          music=1;
        else
          music=0;
      }
      if(button2==0)
        music_now=music;
    } 
    uLCD.printf("\n end \n");
  }
  

}