/*
 -------------------------------------------------------------
  Copyright (c) Microsoft Corporation.  All rights reserved.
 -------------------------------------------------------------
 */
 import * as tf from '@tensorflow/tfjs-node';
 import { promises as fs, readFileSync } from 'fs';


 class ImageModel {
   signature: any;
   modelPath: string;
   height: number;
   width: number;
   inputName: string;
   outputName: string;
   outputKey = "Confidences";
   classes: string[];
   model?: tf.LayersModel;
   exportModelVersion: number;
   latestModelVersion: number;

   constructor(signaturePath: string) {
     const signatureData = readFileSync(signaturePath, "utf8");
     this.signature = JSON.parse(signatureData);
     this.modelPath = `file://../${this.signature.filename}`;
     this.inputName = Object.keys(this.signature.inputs)[0];
     [this.width, this.height] = this.signature.inputs[this.inputName].shape.slice(1,3);
     this.outputName = this.signature.outputs[this.outputKey].name;
     this.classes = this.signature.classes.Label;
     this.exportModelVersion = this.signature.export_model_version;
     this.latestModelVersion = 1;
   }

   checkVersion(){
     if (this.exportModelVersion !== this.latestModelVersion) {
       console.log("Exported model version doesn't match this sample. If there are issues, please export a new model");
     }
   }

   async load() {
     // check if this code is optimized for the model given
     this.checkVersion()
     this.model = await tf.loadLayersModel(this.modelPath);
   }

   dispose() {
     if (this.model) {
       this.model.dispose();
     }
   }

   async predict(image: tf.Tensor) {
       /*
       Given an input image decoded by tensorflow as a tensor,
       preprocess the image into pixel values of [0,1], center crop to a square
       and resize to the image input size, then run the prediction!
        */
       if(!!this.model){
         const confidencesTensor = tf.tidy(() => {

         const [imgHeight, imgWidth] = image.shape.slice(0,2);
         // convert image to from 0 to 255 -> -1 to 1
         const normalizedImage = tf.div(image, tf.scalar(127.5));
         const shiftedImage = tf.sub(normalizedImage, tf.scalar(1))
         // make into a batch of 1 so it is shaped [1, height, width, 3]
         const reshapedImage: tf.Tensor4D = shiftedImage.reshape([1, ...shiftedImage.shape]);
         // center crop and resize
         let top = 0;
         let left = 0;
         let bottom = 1;
         let right = 1;
         if (imgHeight != imgWidth) {
           // the crops are normalized 0-1 percentage of the image dimension
           const size = Math.min(imgHeight, imgWidth);
           left = (imgWidth - size) / 2 / imgWidth;
           top = (imgHeight - size) / 2 / imgHeight;
           right = (imgWidth + size) / 2 / imgWidth;
           bottom = (imgHeight + size) / 2 / imgHeight;
         }
         const croppedImage = tf.image.cropAndResize(
           reshapedImage, [[top, left, bottom, right]], [0], [this.height, this.width]
         );
         if (this.model) {
           return this.model.predict(croppedImage);
         }
       }) as (tf.Tensor | undefined);


       if (confidencesTensor) {
       const confidencesArray = await confidencesTensor.data();
       console.log(confidencesArray)
       confidencesTensor.dispose();
       const output = []
       for (let i=0; i < this.classes.length; i++) {
         output.push({label: this.classes[i], confidence: confidencesArray[i]})

       }
       output.sort((a,b) => {return a.confidence > b.confidence ? -1: 1})
       return {predictions: output};
       }
       } else {
         console.error("Model not loaded, please await this.load() first.");
     }
   }
 }

 async function main(imgPath: string) {
   // read the file from the input path
   const image = await fs.readFile(imgPath);
   // decode the image into a tensor
   const decodedImage = tf.node.decodeImage(image, 3);
   // create and load our model
   const model = new ImageModel('../signature.json');
   await model.load();
   // run the prediction
   const results = await model.predict(decodedImage);
   console.log(results);
   // cleanup
   model.dispose();
 }


 const args = process.argv.slice(2);
 if (args.length !== 1) {
   console.log(`Please specify one argument - the path to your image. Found args: ${args}`);
   process.exit();
 }
 const imgPath = args[0];
 main(imgPath);
