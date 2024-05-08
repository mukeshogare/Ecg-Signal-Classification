DatastPath = 'F:\MajorProject\physionet_ECG_data-main\ecgdataset\';
images = imageDatastore(DatastPath,'IncludeSubfolders',true,'LabelSource','foldernames');
numTrainFiles = 650;
[TrainImage,TestImage] = splitEachLabel(images,numTrainFiles,'randomize');


net = alexnet;
layersTransfer = net.Layers(1:end-3);
numClasses = 3;
layers = [layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Stochastic Gradient Descent with Momentum (SGDM)
options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',8, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',TestImage, ...
    'ValidationFrequency',10, ...
    'Verbose',false, ...
    'Plots', ...
    'training-progress');

netTransfer = trainNetwork(TrainImage,layers,options);

YPred = classify(netTransfer,TestImage);
YValidation = TestImage.Labels;
accuracy = sum(YPred == YValidation) / numel(YValidation);

% Calculate confusion matrix
confMat = confusionmat(YValidation, YPred);

% Calculate performance metrics
TP = confMat(1,1);
FP = confMat(2,1) + confMat(3,1);
FN = confMat(1,2) + confMat(1,3);
TN = confMat(2,2) + confMat(3,3);

PPV = TP / (TP + FP); % Positive Predictive Value
sensitivity = TP / (TP + FN);
F1_score = 2 * ((PPV * sensitivity) / (PPV + sensitivity));

disp(['Accuracy: ', num2str(accuracy)]);
disp(['Positive Predictive Value (PPV): ', num2str(PPV)]);
disp(['Sensitivity: ', num2str(sensitivity)]);
disp(['F1 Score: ', num2str(F1_score)]);

% Plot confusion matrix
plotconfusion(YValidation,YPred);

