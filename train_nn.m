%  Neural network training code 
% Author: Xiujiao Gao
% Parameters

t1 = cputime;
%  number of hidden units
M = 30;
learnrate1 = 0.0005;
learnrate2 = 0.00005;
% learnrate1 = rand(513,M);
% learnrate2 = rand(M+1,10);

% Error Rate Limit, if big than this, continue
err_limit = 0.001;
% Thresh Limit, if err change bigger than it , continues
thresh_limit = 0.001;
% iter num
N = 0;
% iter limit,since I use "or" opertion for while, when iter numer is more
% than N_limit, but the error change a lot, it will keep iterating, this
% N_limit will work when the error is already acceptable
N_limit = 600;
% W1 and W2 first initialize it as ones, then update it during train
% W1 = ones(513,M);
% W2 = ones(M+1,10);
% W1 and W2  initialize it use random numbers between 0 and 1
W1 = rand(513,M)*2 - 1;
W2 = rand(M+1,10)*2 -1;
% W1 = rand(513,M);
% W2 = rand(M+1,10);

fid=fopen('trainout_nn.txt','w');
fprintf(fid,'initialize W1 = \r\n');
for i=1:513
  for j =1:M
  fprintf(fid,'%6.6f \t',W1(i,j));
  end
  fprintf(fid,'\n');
end
fprintf(fid,'initialize W2 = \r\n');
for i=1:M
  for j =1:10
  fprintf(fid,'%6.6f \t',W2(i,j));
  end
  fprintf(fid,'\n');
end
        
% X, feature matrix 
X = load(strcat('train0.txt'));
[Xrows,Xcolumns] = size(X);
% get train data , the left will be used as validation data
X(ceil(Xrows*0.95)+1:Xrows,:)=[];
[Xrows,Xcolumns] = size(X);
xones = ones(Xrows,1);
X = [xones X];
% T, lable for each x feature,is 1 of 10 vector
T = zeros(Xrows,10);
T(:,1) = 1;
for i=1:9    
    x = load(strcat('train',num2str(i),'.txt'));
    [xrows,xcolumns] = size(x);
    % get training data 
    x(ceil(xrows*0.95)+1:xrows,:)=[];
    [xrows,xcolumns] = size(x);
    xones = ones(xrows,1);
    X = [X; xones x];
    t = zeros(xrows,10);
    t(:,i+1) = 1;
    T = [T;t];
end

A = X*W1;
[Arows,Acolumns] = size(A);
Z = zeros(Arows,Acolumns);
for i = 1:Arows
    for j = 1:Acolumns
        Z(i,j) = tanh(A(i,j));
    end
end

z = ones(Arows,1)*1.0;
Z = [z Z];
R = Z*W2;
% compute new Y
Sig = exp(R);
[Sigrows,Sigcolumns] = size(Sig);
% Get new Y
rowsum = sum(Sig,2);
temp = zeros(Sigrows,Sigcolumns);
for i = 1:Sigcolumns
    temp(:,i) = rowsum;
end
Y = Sig./temp;
% get max value from Y for each row and the corresponding column number
[y,n] = max(Y');
Y_Lable = zeros(Sigrows,Sigcolumns);
for i= 1:Sigrows
    Y_Lable(i,n(i)) = 1;
end
% get error rate
E = xor(Y_Lable,T);
err = (sum(sum(E))/2)/Sigrows;

% update W
thresh = 1;
while(thresh > thresh_limit | err > err_limit | N < N_limit) % threshhold of updating
    gradientW1 = zeros(513,M);
    gradientW2 = zeros(M+1,10);
%     Y_T = T.*(ones(size(Y))-Y);
    Y_T = Y-T;
    
    gradientW2 =  Z'*Y_T;
    W2 = W2 - learnrate2.*gradientW2;
    W2_copy = W2;
    W2_copy(1,:) = [];
    Z_copy = Z;
    Z_copy(:,1) = [];
    Q = Y_T*W2_copy';
    B = Q.*(ones(size(Z_copy))-Z_copy.^2);
    gradientW1 = X'*B;
% get new W1 and W2

W1 = W1 - learnrate1.*gradientW1;
% Get new Y
A = X*W1;
[Arows,Acolumns] = size(A);
Z = zeros(Arows,Acolumns);
for i = 1:Arows
    for j = 1:Acolumns
        Z(i,j) = tanh(A(i,j));
    end
end

z = ones(Arows,1)*1.0;

Z = [z Z];
R = Z*W2;
% compute new Y
Sig = exp(R);
% Get new Y
rowsum = sum(Sig,2);
temp = zeros(size(Sig));
for i = 1:Sigcolumns
    temp(:,i) = rowsum;
end
Y = Sig./temp;

% get max value from Y for each row and the corresponding column number
[y,n] = max(Y');
Y_Label = zeros(Sigrows,Sigcolumns);
for i= 1:Sigrows
    Y_Label(i,n(i)) = 1;
end
% get error rate
E = xor(Y_Label,T);
err_old = err;
err = (sum(sum(E))/2)/Sigrows
thresh = err_old - err;
if (thresh < 0)
fprintf('Error rate increase from %6.6f to %6.6f, the learning rate maybe too large',err_old,err);
end
N = N+1
end     
t2 = cputime;

fprintf(fid,'time = % 6.6f \r\n',t2-t1);
fprintf(fid,'err_limit = % 6.6f \r\n',err_limit);
fprintf(fid,'err_old = % 6.6f \r\n',err_old);
fprintf(fid,'err = % 6.6f \r\n',err);
fprintf(fid,'thresh_limit = % 6.6f \r\n',thresh_limit);
fprintf(fid,'thresh = % 6.6f \r\n',thresh);
fprintf(fid,'learnrate1 = % 6.6f \r\n',learnrate1);
fprintf(fid,'learnrate2 = % 6.6f \r\n',learnrate2);
fprintf(fid,'iter times = % d \r\n',N);
fidw1=fopen('W1.txt','w');
fidw2=fopen('W2.txt','w');
fprintf(fid,'final W1 = \r\n');
for i=1:513
  for j =1:M
  fprintf(fid,'%6.6f \t',W1(i,j));
  fprintf(fidw1,'%6.6f \t',W1(i,j));
  end
  fprintf(fid,'\n');
  fprintf(fidw1,'\n');
end
fprintf(fid,'final W2 = \r\n');
for i=1:M+1
  for j =1:10
  fprintf(fid,'%6.6f \t',W2(i,j));
  fprintf(fidw2,'%6.6f \t',W2(i,j));
  end
  fprintf(fid,'\n');
  fprintf(fidw2,'\n');
end


for i=1:Sigrows
  for j =1:Sigcolumns
  fprintf(fid,'%d \t',Y_Label(i,j));
  end
  fprintf(fid,'\n');
end
        
fclose(fid);
fclose(fidw1);
fclose(fidw2);






