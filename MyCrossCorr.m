classdef MyCrossCorr < nnet.layer.Layer ...
        & nnet.layer.Acceleratable
    % Example custom weighted addition layer.

    properties (Learnable)
        % Layer learnable parameters
            
        % Scaling coefficients
        Weights
    end
    
    methods
        function layer = MyCrossCorr(numInputs,name) 
            % layer = weightedAdditionLayer(numInputs,name) creates a
            % weighted addition layer and specifies the number of inputs
            % and the layer name.

            % Set number of inputs.
            layer.NumInputs = numInputs;

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Weighted addition of " + numInputs +  ... 
                " inputs";
        
            % Initialize layer weights.
            layer.Weights = rand(1,numInputs); 
        end
        
        function Z = predict(layer, varargin)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.

            %%Original
%             X = varargin;
%             W = layer.Weights;
%             
%             % Initialize output
%             X1 = X{1};
%             sz = size(X1);
%             Z = zeros(sz,'like',X1);
%             
%             % Weighted addition
%             for i = 1:layer.NumInputs
%                 Z = Z + W(i)*X{i};
%             end
             %%Original
            
            X = varargin;
            W = layer.Weights;
            
            % Initialize output
            X1 = X{1};
            X2 = X{2};

            c = ones([1 1 1 1]);

            Sz1=zeros(size(X1));
            Sz1(:,:,:,:) = X1;
            Sx1=zeros(size(X2));
            Sx1(:,:,:,:) = X2;    
            
%             Sz1 = dlarray(Sz1);
%             Sx1 = dlarray(Sx1);
            
      

            Z = cross_corr(Sz1, Sx1, c);
            Z = cast(Z, "like", X1);
            %Z=dlarray(Z);

%             if class(X1(1,1,1,1))==single
%                 
%             end
            %class(Z)
% 
%             sz = size(Z);
%             Z = zeros(sz,'like',Z) + Z;
            % Weighted addition
%             for i = 1:layer.NumInputs
%                 Z = Z + W(i)*X{i};
%             end
        end
    end
end