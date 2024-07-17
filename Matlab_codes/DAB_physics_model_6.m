

% % Define parameter ranges
% Ll1_range = 2.975:1.2:8.925;         % Example range for Ll1 in microhenries
% Ll2_range = 1.425:0.57:4.275;         % Example range for Ll2 in microhenries
% Ron1_range = 15:30:150;        % Example range for Ron1 in milliohms
% Ron2_range = 4:8:40;        % Example range for Ron2 in milliohms
% delta1_range = 0:0.2:pi/2;   % Example range for delta1 in radians
% delta2_range = 0:0.2:pi/2;   % Example range for delta2 in radians
% phi_range = 0:0.1:pi/2;      % Example range for phi in radians
% 
% % Preallocate the dataset matrix
% num_combinations = length(Ll1_range) * length(Ll2_range) * length(Ron1_range) * ...
%                    length(Ron2_range) * length(delta1_range) * length(delta2_range) * length(phi_range);

% 
% % Counter for indexing dataset
% count = 1;

% Number of samples to generate
num_samples = 100000;
dataset = zeros(num_samples, 13);
% Define the number of distinct values
num_distinct_L = 25;
num_distinct_R = 25;
num_distinct_phi = 20;
num_distinct_delta = 20;

% Define the ranges for each parameter
L1_range = linspace(2.975, 8.925, num_distinct_L);
L2_range = linspace(1.425, 4.275, num_distinct_L);
R1_range = linspace(15, 150, num_distinct_R);
R2_range = linspace(4, 100, num_distinct_R);
phi_range = linspace(0, pi/2, num_distinct_phi);
delta1_range = linspace(0, pi/2, num_distinct_delta);
delta2_range = linspace(0, pi/2, num_distinct_delta);

% Define the values for V1 and V2 (since they have only 1 value, they are constants)
Vin = 160; % Example value for V1
Vout = 120; % Example value for V2

% Pre-allocate space for the samples and results
L1_samples = zeros(num_samples, 1);
L2_samples = zeros(num_samples, 1);
R1_samples = zeros(num_samples, 1);
R2_samples = zeros(num_samples, 1);
phi_samples = zeros(num_samples, 1);
delta1_samples = zeros(num_samples, 1);
delta2_samples = zeros(num_samples, 1);
results = zeros(num_samples, 13); % Adjust the size and type based on the output of dab_physics

% Randomly sample from the distinct values
for i = 1:num_samples
    L1_samples(i) = L1_range(randi(num_distinct_L));
    L2_samples(i) = L2_range(randi(num_distinct_L));
    R1_samples(i) = R1_range(randi(num_distinct_R));
    R2_samples(i) = R2_range(randi(num_distinct_R));
    phi_samples(i) = phi_range(randi(num_distinct_phi));
    delta1_samples(i) = delta1_range(randi(num_distinct_delta));
    delta2_samples(i) = delta2_range(randi(num_distinct_delta));
    
    fprintf("%f\n",i);
    [results] = DAB_physics_model(Vin, Vout, phi_samples(i), delta1_samples(i), delta2_samples(i), R1_samples(i), R2_samples(i),L1_samples(i), L2_samples(i));

                            % Store parameters and results in the dataset
    dataset(i, :) = [phi_samples(i), delta1_samples(i), delta2_samples(i), R1_samples(i), R2_samples(i),L1_samples(i), L2_samples(i),results];

    
end
function v_y = calculate_vth(i, wt, phi, delta1, delta2, V1, V2)
    if i == 1
        phi_y = 0; delta_y = delta1; Vy = V1;
    else
        phi_y = phi; delta_y = delta2; Vy = V2;
    end

    if (phi_y - delta_y) < wt && wt < (phi_y + delta_y)
            v_y = 0;
        elseif (phi_y + delta_y) < wt && wt < (pi + phi_y - delta_y)
            v_y = Vy;
        elseif (pi + phi_y - delta_y) < wt && wt < (pi + phi_y + delta_y)
            v_y = 0;
        elseif (pi + phi_y + delta_y) < wt && wt < (2 * pi + phi_y - delta_y)
            v_y = -Vy;
        elseif (phi_y - (pi - delta_y)) < wt && wt < (phi_y - delta_y)
            v_y = -Vy;
        else
        fprintf("%s","error");
        v_y = 0;  % or any default value    

    end
end 

% Loop over all parameter combinations
% for Ll1 = Ll1_range
%     for Ll2 = Ll2_range
%         for Ron1 = Ron1_range
%             for Ron2 = Ron2_range
%                 for delta1 = delta1_range
%                     for delta2 = delta2_range
%                         for phi = phi_range
%                             % Set input parameters
%                             Vin = 160;    % Example input voltage
%                             Vout = 120;   % Example output voltage
% 
%                             % Call the model function
%                             count = count + 1;
%                             fprintf("%f\n",count);
%                         end
%                     end
%                 end
%             end
%         end
%     end
% end

dataset_table = array2table(dataset, 'VariableNames', ...
    {'Ll1', 'Ll2', 'Ron1', 'Ron2', 'delta1', 'delta2', 'phi', 'Pdc_in', 'I1_RMS','I2_RMS','Pdc_out', 'Pdc_in_minus_Pdc_out', 'P_loss_total'});

% Save the table to a CSV file
writetable(dataset_table, 'DAB_dataset_new.csv');


function [y] = DAB_physics_model(Vin, Vout, phi, delta1, delta2, Ron1, Ron2,Ll1,Ll2)


%Ron1= 15; in mohm 
%Ron2 = 4; in mohm


V1 = Vin; 
n1 = 7; 
n2 = 5;
V2 = n1/n2 * Vout; 

x = 0.16;
fsw = 1e5;

% Ll1 = 5.96 µH
% Ll2 =2.85µH
% Lm = 776.890112 µH

% Define constants
Ron1 = Ron1*1e-3; %17e-3; 
n1 = 7; % Example value, adjust as needed
n2 = 5; % Example value, adjust as needed
Ron2 = (n1/n2)^2 * Ron2*1e-3; % 5e-3;
Rl1 = 20e-3;
Rl2 = (n1/n2)^2 * 15e-3;
L1 = (Ll1 +x) * 1e-6; % Example value, adjust as needed
L2 = (n1/n2)^2 * (Ll2 +x*((n2/n1)^2)) * 1e-6; % Example value, adjust as needed
Lm = 776.89e-6; % Example value, adjust as needed
Rm = 1.0805e+07; 

esr_Cb1 = 5e-3; 
esr_Cb2 = (n1/n2)^2 *5e-3; 

w = 2 * pi * fsw; % Angular frequency

% Time vector from 0 to 10 microseconds
t = linspace(0, 10e-6, 100);
phase1 = [delta1 -delta1]; 
phase_pk = [delta1 -delta1 phi+delta2 phi-delta2];
phase2 = [phi+delta2 phi-delta2]; 
% Preallocate arrays
Z1 = zeros(1, 26);
Z2 = zeros(1, 26);
Z3 = zeros(1, 26);
Ztot = zeros(1, 26);
Z12 = zeros(1, 26);
Z23 = zeros(1, 26);
Z13 = zeros(1, 26);
v1_abs = zeros(1, 26);
v2_abs = zeros(1, 26);
Ai1 = zeros(1, 26);
Bi1 = zeros(1, 26);
Ai2 = zeros(1, 26);
Bi2 = zeros(1, 26);
i1_t = zeros(length(t), 26);
i2_t = zeros(length(t), 26);
v1_t = zeros(length(t), 26);
v2_t = zeros(length(t), 26);
i1_phase = zeros(length(phase1), 26);
i2_phase = zeros(length(phase2), 26);
i1_pk = zeros(length(phase_pk), 26);
i2_pk = zeros(length(phase_pk), 26);

% Initialize sums
sum_i1_RMS_squared = 0;
sum_i2_RMS_squared = 0;
sum_Pac_1 = 0;
sum_Pac_2 = 0;
i1_abs = zeros(1, 26);
angle_i1 = zeros(1, 26);
angle_i2 = zeros(1, 26);
cos_angle = zeros(1, 26);

% Perform calculations
for k = 1:2:53
    k_idx = (k + 1) / 2;
    Z1(k_idx) = 2 * Ron1 + Rl1 + esr_Cb1 + 1i * (2 * pi * fsw * k * L1);
    Z2(k_idx) = 2 * Ron2 + Rl2 + esr_Cb2 + 1i * (2 * pi * fsw * k * L2);
    Z3(k_idx) = Rm * 1i * (2 * pi * fsw * k * Lm)/(Rm + 1i * (2 * pi * fsw * k * Lm));
    Ztot(k_idx) = Z1(k_idx) * Z2(k_idx) + Z1(k_idx) * Z3(k_idx) + Z2(k_idx) * Z3(k_idx);
    Z12(k_idx) = Ztot(k_idx) / Z3(k_idx);
    Z23(k_idx) = Ztot(k_idx) / Z1(k_idx);
    Z13(k_idx) = Ztot(k_idx) / Z2(k_idx);

    v1_abs(k_idx) = 4 .* V1 .* cos(k .* delta1) / (k * pi);
    v2_abs(k_idx) = 4 * V2 * cos(k * delta2) / (k * pi);

    Ai1(k_idx) = (v1_abs(k_idx) / abs(Z12(k_idx))) * cos(angle(Z12(k_idx))) - ...
                 (v2_abs(k_idx) / abs(Z12(k_idx))) * cos(k * phi + angle(Z12(k_idx))) + ...
                 (v1_abs(k_idx) / abs(Z13(k_idx))) * cos(angle(Z13(k_idx)));

    Bi1(k_idx) = -(v1_abs(k_idx) / abs(Z12(k_idx))) * sin(angle(Z12(k_idx))) + ...
                 (v2_abs(k_idx) / abs(Z12(k_idx))) * sin(k * phi + angle(Z12(k_idx))) - ...
                 (v1_abs(k_idx) / abs(Z13(k_idx))) * sin(angle(Z13(k_idx)));

    Ai2(k_idx) = -(v1_abs(k_idx) / abs(Z12(k_idx))) * cos(angle(Z12(k_idx))) + ...
                 (v2_abs(k_idx) / abs(Z12(k_idx))) * cos(k * phi + angle(Z12(k_idx))) + ...
                 (v2_abs(k_idx) / abs(Z23(k_idx))) * cos(k*phi + angle(Z23(k_idx)));

    Bi2(k_idx) = +(v1_abs(k_idx) / abs(Z12(k_idx))) * sin(angle(Z12(k_idx))) - ...
                 (v2_abs(k_idx) / abs(Z12(k_idx))) * sin(k * phi + angle(Z12(k_idx))) - ...
                 (v2_abs(k_idx) / abs(Z23(k_idx))) * sin(k*phi + angle(Z23(k_idx)));

    i1_t(:, k_idx) = Ai1(k_idx) * sin(k * w * t) + Bi1(k_idx) * cos(k * w * t);
    i2_t(:, k_idx) = Ai2(k_idx) * sin(k * w * t) + Bi2(k_idx) * cos(k * w * t);

    Vm_t(:, k_idx) = 2 * pi * fsw * k * Lm * ((Ai1(k_idx)+Ai2(k_idx)) * cos(k * w * t) - (Bi1(k_idx)+Bi2(k_idx)) * sin(k * w * t));

    i1_phase(:, k_idx) = Ai1(k_idx) * sin(k * phase1) + Bi1(k_idx) * cos(k * phase1);
    i2_phase(:, k_idx) = Ai2(k_idx) * sin(k * phase2) + Bi2(k_idx) * cos(k * phase2);

    i1_pk(:, k_idx) = Ai1(k_idx) * sin(k * phase_pk) + Bi1(k_idx) * cos(k * phase_pk);
    i2_pk(:, k_idx) = Ai2(k_idx) * sin(k * phase_pk) + Bi2(k_idx) * cos(k * phase_pk);

    v1_t(:, k_idx) = v1_abs(k_idx) * sin(k * w * t);
    v2_t(:, k_idx) = v2_abs(k_idx) * sin(k * (w * t - phi));

    i1_abs(k_idx) = sqrt((Ai1(k_idx)^2 + Bi1(k_idx)^2));

    angle_i1(k_idx) = atan2(Bi1(k_idx) , Ai1(k_idx));
    
    angle_i2(k_idx) = k*phi + atan2(Bi2(k_idx) , Ai2(k_idx));

    cos_angle (k_idx) = cos(angle_i1(k_idx));

    sum_Pac_1 = sum_Pac_1 + (v1_abs(k_idx) / sqrt(2)) * sqrt((Ai1(k_idx)^2 + Bi1(k_idx)^2) / 2) * cos(angle_i1(k_idx));

    sum_i1_RMS_squared = sum_i1_RMS_squared + (Ai1(k_idx)^2 + Bi1(k_idx)^2);
    sum_i2_RMS_squared = sum_i2_RMS_squared + (Ai2(k_idx)^2 + Bi2(k_idx)^2);
    
    sum_Pac_2 = sum_Pac_2 + (v2_abs(k_idx) / sqrt(2)) * (sqrt((Ai2(k_idx)^2 + Bi2(k_idx)^2) / 2)) * cos(angle_i2(k_idx));

end

I1_RMS = sqrt(sum_i1_RMS_squared / 2);
I2_RMS = sqrt(sum_i2_RMS_squared / 2) * (7/5);
Pac_1 = sum_Pac_1;
Pac_2 = sum_Pac_2;
%fprintf("%f\n",Pac_1);
%fprintf("%f\n",Pac_2);
% %Sum all the harmonics to get the total current
% i1_total_t = sum(i1_t, 2);
% i2_total_t = sum(i2_t, 2);
% v1_total_t = sum(v1_t, 2);
% v2_total_t = sum(v2_t, 2);

Vm_total_t = sum(Vm_t, 2);
%fprintf('%f\n', Vm_total_t)

% Calculate the positive area under the Vm curve (volt-second)
positive_Vm = Vm_total_t;
positive_Vm(positive_Vm < 0) = 0; % Set negative values to 0
volt_sec = trapz(t, positive_Vm);

% Display the result
%disp(['The volt-second (positive area under Vm curve) is: ', num2str(volt_sec), ' V·s']);

i1_sw_curr = sum(i1_phase,2);
i2_sw_curr = sum(i2_phase,2);
%fprintf("%f\n",i1_sw_curr);
i1_sw_pk_curr = sum(i1_pk,2);
i2_sw_pk_curr = sum(i2_pk,2);
i1_max = max(abs(i1_sw_pk_curr));
i2_max = 7/5 * max(abs(i2_sw_pk_curr));

sw_on_curr_array = [i1_sw_curr; i2_sw_curr]';
%fprintf("%f\n",sw_on_curr_array);

% Calculate and display an additional result
L = L1 + L2;
P_lossless = V1 * V2 * phi * (1 - phi / pi) / (2 * pi * fsw * L);

Rds_on =[17 5]*1e-3;
Coss = [1200 1200]*1e-12;
Rg_on=[10 10];
Rg_off=[5 5];
Cin=1e-12*[1200 2000];
Vdr=[6 5];
Vpl=[3 2.3];
Vth=[1.7 1.1];
Cgd=1e-12*[14 10];
Qoss = [240 250]*1e-9;
Qg = [30 36]*1e-9;

%%
N_tot =2;
SW_ON_mat_redefind = zeros(2, N_tot);
zvs = zeros(2, N_tot);
Vth_mat_redefind = [ calculate_vth(2, delta1, phi, delta1, delta2, V1, V2) calculate_vth(1, phi+delta2, phi, delta1, delta2, V1, V2); calculate_vth(2, pi-delta1, phi, delta1, delta2, V1, V2) calculate_vth(1, phi+pi-delta2, phi, delta1, delta2, V1, V2)];

for i = 1:2
    SW_ON_mat_redefind(:, i) = [sw_on_curr_array(2*i-1) ; sw_on_curr_array(2*i)];
end
%fprintf("a%f\n",SW_ON_mat_redefind)
%disp(SW_ON_mat_redefind);
V = [V1 V2]; 
N = [7 5];
L = L1 +L2 ; 

delta = [delta1 delta2];
for i=1:2
if delta(i) < 0.01
    if SW_ON_mat_redefind(1,i)<0
        ind_energy = 0.5*L*(SW_ON_mat_redefind(1,i))^2;
        cap_energy = -2*Coss(i)*((N(i)/N(1))^2)*Vth_mat_redefind(1,i)*V(i);

        if  ind_energy >= cap_energy
            zvs(1,i)=1; zvs(2,i)=1;
        else
            zvs(1,i) = 1 - sqrt((cap_energy - ind_energy)/cap_energy); 
            
            zvs(2,i) = zvs(1,i);
        end
    else
        zvs(1,i)=0; zvs(2,i)=0;
    end
else
    for k=1:2
        if SW_ON_mat_redefind(k,i)<0
            ind_energy = 0.5*L*(SW_ON_mat_redefind(k,i))^2;
            cap_energy = -2*Coss(i)*((N(i)/N(1))^2)*Vth_mat_redefind(k,i)*V(i) + (-1)^(k+1)*Coss(i)*((N(i)/N(1))^2)*(V(i))^2;
            if ind_energy >= cap_energy
                zvs(k,i)=1;
            else
                zvs(k,i) = 1 - sqrt((cap_energy-ind_energy)/cap_energy);
            end
        else
            zvs(k,i)=0;
        end
    end
end    
end


t1=zeros(1,N_tot);
t2=zeros(1,N_tot);
t3=zeros(1,N_tot);
t4=zeros(1,N_tot);
sw_loss = zeros(2, N_tot);
Coss_loss_dist = zeros (1, N_tot);
gate_drive_loss_dist = zeros (1, N_tot);


for i=1:N_tot
    Coss_loss_dist(i) = 4*0.5*Qoss(i)*V(i)*N(i)/N(1)*fsw;
    gate_drive_loss_dist(i) = 4*Qg(i)*Vdr(i)*fsw;
    t3(i)=Cgd(i)*(1+Rg_off(i))*V(i)*N(i)/N(1)/Vpl(i);
    t4(i)=(1+Rg_off(i))*Cin(i)*log(Vpl(i)/Vth(i));
    t1(i) = (1+Rg_on(i))*Cin(i)*log((Vdr(i)-Vth(i))/(Vdr(i)-Vpl(i)));
    t2(i) = (1+Rg_on(i))*Cgd(i)*V(i)*N(i)/N(1)/(Vdr(i)-Vpl(i));
    for k = 1:2
        sw_loss(k,i)=2*V(i)*N(i)/N(1)*abs(SW_ON_mat_redefind(k,i))*fsw*(t3(i)+t4(i));
        
        sw_loss(k,i) = sw_loss(k,i) + (1-zvs(k,i))*(2*V(i)*N(i)/N(1)*abs(SW_ON_mat_redefind(k,i))*fsw*(t1(i)+t2(i)) + ...
                        2*Qoss(i)*V(i)*N(i)/N(1)*fsw ); %%% added 2A current
        
    end
end
sw_loss_dist=sum(sw_loss);

newzvs = zvs(1:2, :);


%% Magnetic Losses


Ae_1 = 2* 135 * 1e-6; % m^2
L_value_1 = L1 - x * 1e-6; % H
N_turns_1 = 4; 
Bm_1 = (i1_max *L_value_1) /(N_turns_1*Ae_1); % in T
Ve_1 = 5750*2; %in mm^3

Ae_2 = 135 * 1e-6; % m^2
L_value_2 = (n2/n1)^2 * (L2 - x * 1e-6 ); % H
N_turns_2 = 4; 
Bm_2 = (i2_max *L_value_2) /(N_turns_2*Ae_2); % in T
Ve_2 = 5750; %in mm^3


% Coefficients for material R
a = 2.67;
b = -3.42e-2;
c = 1.75e-4;
d = 0;
e = 0;

% Input temperature T (in degrees Celsius)
T = 40; %input('Enter the temperature T (in degrees Celsius): ');

% Calculate Tc
Tc = a + b*T + c*T^2 + d*T^3 + e*T^4;

% Coefficients for material R at 100°C
% Define a, c, and d for different frequency ranges
coeffs = [
    0.074, 1.43, 2.85; % f < 100 kHz
    0.036, 1.7, 2.68; % 100 kHz <= f < 500 kHz
    0.014, 1.84, 2.2   % f >= 500 kHz
];

% Define the frequency ranges
freq_ranges = [
    0, 99;    % f < 100 kHz
    100, 499;  % 100 kHz <= f < 500 kHz
    500, inf   % f >= 500 kHz
];

% Input parameters
f = 100; %input('Enter the frequency f (in kHz): ');
B_1 = 10 * Bm_1 ; %input('Enter the magnetic flux density B (in kG): ');
B_2 = 10 * Bm_2 ;

% Determine the correct coefficients based on the input frequency
if f < 100
    a = coeffs(1, 1);
    c = coeffs(1, 2);
    d = coeffs(1, 3);
elseif f >= 100 && f < 500
    a = coeffs(2, 1);
    c = coeffs(2, 2);
    d = coeffs(2, 3);
else
    a = coeffs(3, 1);
    c = coeffs(3, 2);
    d = coeffs(3, 3);
end


% Calculate the power loss
Pcore_L1 = a * (f^c) * (B_1^d) *10^-6 * Ve_1 *Tc;
Pcore_L2 = a * (f^c) * (B_2^d) *10^-6 * Ve_2 *Tc;

Np =7; 
Ae_lm = 2* 194 *1e-6; % in m^2
B_lm = volt_sec/(2*Np*Ae_lm); % in T
Ve_lm = 2* 10200; % in mm^3
Pcore_Lm = a * (f^c) * ((B_lm*10)^d) *10^-6 * Ve_lm *Tc;

%%


P_Cond = [I1_RMS^2 * (2 * Ron1 + Rl1 +esr_Cb1)  (5/7*I2_RMS)^2 * (2 * Ron2 + Rl2 + esr_Cb2)];
% newzvs = newzvs(:).'
% 
% sw_on_curr_array
% sw_loss
% sw_loss_dist
% Pcore_L1
% Pcore_L2
% Pcore_Lm
% 

P_loss_total = sum(P_Cond) + sum(sw_loss_dist) + Pcore_L1 + Pcore_L2 + Pcore_Lm;
% 

Pdc_in = Pac_1 + sw_loss_dist(1)  ; 
Pdc_out = -Pac_2 - sw_loss_dist(2) - Pcore_L2 -Pcore_L1 - Pcore_Lm;

y = [Pdc_in Pdc_out Pdc_in-Pdc_out P_loss_total,I1_RMS,I2_RMS]; 
return
end











