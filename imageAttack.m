
% jpeg_params = [30,40,50,60,70,80,90,100];%8
% scale_params = [0.5,0.75,0.9,1.1,1.5,2.0];%6
% %speckle noise, salt and pepper noise
% noise_params = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01];%10
% %brightness and contrast adjustment
% adjust_params = [-20,-10,10,20];%4
% gamma_params = [0.75, 0.9, 1.1, 1.25];
% gauss_params = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
% rotation_params = [-1,-2,-3,-4,-5,1,2,3,4,5];
% 
% % Root = "/home/xumengqi/Downloads/multimedia/databases/Standard benchmark image/";
% Root = "/home/xumengqi/Downloads/multimedia/databases/Copydays/";
% 
% % File = dir(fullfile(Root+"ORIGINAL_IMAGES/", '*.png'));
% File = dir(fullfile(Root+"ORIGINAL_IMAGES/", '*.jpg'));
% filenames = {File.name}';
% for i=1 :length(filenames)
%     
%     image = imread(Root +"ORIGINAL_IMAGES/"+ filenames(i));
% %     iname = strrep(filenames(i), ".png", "");
%     iname = strrep(filenames(i), ".jpg", "");
% % 
%     ppath = Root + "JPEG_compression/"+ iname;
%     mkdir(ppath);
% 
%     for j=1 :length(jpeg_params)
%         param = jpeg_params(j);
%         writepath = ppath + "/Quality_factor_" +num2str(param) + ".jpg";
%         disp(writepath);
%         imwrite(image, writepath, 'jpg','Quality', param);  
%     end
% 
%     ppath = Root+"Image_Scaling/"+iname;
%     mkdir(ppath);
%     for j=1 :length(scale_params)
%     	param = scale_params(j);
%     	writepath = ppath+"/Ratio_"+num2str(param)+ ".jpg";
%     	J = imresize(image, param);
%     	imwrite(J, writepath, 'jpg');
%     end
% 
%     ppath1 = Root + "Speckle_noise/"+iname;
%     mkdir(ppath1);
%     ppath2 = Root + "Salt_and_Pepper_noise/"+iname;
%     mkdir(ppath2);
%     for j=1 :length(noise_params)
%     	param = noise_params(j);
%     	writepath1 = ppath1+"/Variance_"+num2str(param)+".jpg";
%     	J1 = imnoise(image, 'speckle', param);
%     	imwrite(J1, writepath1, 'jpg');
% 
%     	writepath2 = ppath2 +"/Density_"+num2str(param)+".jpg";
%     	J2 = imnoise(image, 'salt & pepper',param);
%     	imwrite(J1, writepath2, 'jpg');
%     end
% 
% 
%     ppath = Root + "Gamma_correction/"+iname;
%     mkdir(ppath);
%     for j=1 :length(gamma_params)
%     	param = gamma_params(j);
%     	writepath = ppath+"/gamma_"+num2str(param)+".jpg";
%     	J = imadjust(image,[],[],param);
%     	imwrite(J, writepath, 'jpg');
%     end
% 
%     ppath = Root+"Gauss_filtering/"+iname;
%     mkdir(ppath);
%     for j=1 :length(gauss_params)
%     	param = gauss_params(j);
%     	writepath = ppath+"/Standard_deviation_"+num2str(param)+".jpg";
%     	J = imgaussfilt(image,param);
%     	imwrite(J, writepath, 'jpg');
%     end
% 
%     ppath = Root+"Image_Rotation/"+iname;
%     mkdir(ppath);
%     for j=1 :length(rotation_params)
%     	param = rotation_params(j);
%     	writepath = ppath+"/Rotation_angle_"+num2str(param)+".jpg";
%         J = imrotate(image, param);
%         imwrite(J, writepath, 'jpg');
%     end
%     
% end
% 
% 


path = "/home/xumengqi/Downloads/multimedia/databases/Standard benchmark image/ORIGINAL_IMAGES/airplane.png";
path1 = "/home/xumengqi/Downloads/multimedia/databases/Standard benchmark image/ORIGINAL_IMAGES/airplane_1.png";
path2 = "/home/xumengqi/Downloads/multimedia/databases/Standard benchmark image/ORIGINAL_IMAGES/airplane_5.png";
img = imread(path);
J1 = imrotate(img, 1);
J2 = imrotate(img, 5);
J1 = imresize(J1,[512,512]);
J2 = imresize(J2, [512,512]);
imwrite(J1, path1);
imwrite(J2, path2);


