b = 0;  % Belief
d = 0;  % Disbelief
u = 0;  % Uncertainty
a = 0.5;  % Base rate
w = {b, d, u, a};  % Opinion
W = []; % Vector to store expectation probabilities
r = 10;  % Number of positive past observations for x
s = 10;  % Number of negative spast observations for x'
phi = 0; % fusion factor (weight/importance) 
delta = 0; % Longevity factor: 0 <= delta <= 1
t = 0;  % Current time
t_r = 0; % Time of the recording. Used with longevity factor to forget old records
p_a_b = {r , s}; % Rating of a for b. Where r - positive rating and s - negative rating

alfa = compute_alfa(10, 0.5);
beta = compute_beta(2, 0.5);

X = 0:.09:1; % Obeservable values

y = betapdf(X, alfa, beta);
plot(y);
G = digraph({'A' 'B' 'B'}, ...
            {'B' 'A' 'C'});

w_a_b = {0.9, 0.0, 0.1, 0.5}; 
w_b_d = {0.9, 0.0, 0.1, 0.5};
w_a_c = {0.9, 0.0, 0.1, 0.5};
w_c_d = {0.9, 0.0, 0.1, 0.5};
w_b_c = {0.9, 0.0, 0.1, 0.5}; 

t_a_b_d = transitivity(w_a_b, w_b_c);
t_a_c_d = transitivity(w_a_c, w_c_d);
fused_op_a_d = cummulative_fusion(t_a_b_d, t_a_c_d);
t_a_b_c_d = transitivity(transitivity(w_a_b, w_b_c), w_c_d);

[w_a_b1, w_a_b2] = split_opinion(w_a_b, phi);
[w_c1_d, w_c2_d] = split_opinion(w_c_d, phi);
[w_b1_d, w_b2_d] = split_opinion(w_b_d, phi);
[w_b2_c2, w_b2_c1] = split_opinion(w_b_c, phi);
w_a_c1 = w_a_c;

U = uncertainty_in_path(w_a_b1, w_a_c1, ...
                    w_a_b2, w_c2_d, ...
                    w_b1_d, w_c1_d, ...
                    w_b2_c2, w_b2_d);

u_a_d = uncertainty(U{1}, U{2}, U{3});



%%%
% Function to compute opinion w values
%%%
function w = compute_opinion(r, s, a)
    
    % belief
    b = r / (r + s + 2);

    % disbelief
    d = s / (r + s + 2);

    % uncertainty
    u = 2 / (r + s + 2);

    % new opinion
    w = {b, d, u, a};
end

%%%
% Beta distribution argument functions
%%%
function alfa = compute_alfa(r, a)
    alfa = r + 2*a;
end

function beta = compute_beta(s, a)
    beta = s + 2*(1 - a);
end

%%%
% Computes opinion's w probability expectation
%%%
function exp = expectation(w)
    exp = w{1}+w{4}*w{3}; 
end

%%%
% Transitivity operation from A->B->C
%%%
function trans_opinion = transitivity(w_a_b, w_b_c)
    % Transitive base rate a_t for A = base rate of C 
    a_t = w_b_c{4};
    
    % Transitive uncertainty u_t for 
    % A = disbilief_A_B + uncertainty_A_B + bilief_A_B * uncertainty_B_C
    u_t = w_a_b{2} + w_a_b{3} + w_a_b{1}*w_b_c{3};
    
    % Transitive disbelief d_t for A = bilief_A_B * disbelief_B_C
    d_t = w_a_b{1} * w_b_c{2};

    % Transitive belief b_t for A = bilief_A_B * bilief_B_C
    b_t = w_a_b{1} * w_b_c{1};

    trans_opinion = {b_t, d_t, u_t, a_t};
end

%%%
% Cummulative Fusion.
% Computes a opinion to obtain fused trust betwee w_a_c and w_b_c
%%%
function fused_trust = cummulative_fusion(w_a_c, w_b_c)
    a_fused = w_a_c{4};
    u_fused = (w_a_c{3} * w_b_c{3}) / (w_a_c{3} + w_b_c{3} - w_a_c{3} * w_b_c{3});
    d_fused = (w_a_c{2} * w_b_c{3} + w_a_c{3} * w_b_c{2}) / (w_a_c{3} + w_b_c{3} - w_a_c{3} * w_b_c{3});
    b_fused = (w_a_c{1} * w_b_c{3} + w_b_c{1} * w_a_c{3}) / (w_a_c{3} + w_b_c{3} - w_a_c{3} * w_b_c{3});

    fused_trust = {b_fused, d_fused, u_fused, a_fused};
end

%%%
% Splits a opinion in two separate opinions w_1 and w_2
% w_original = cummmulative_fusion(w_1, w_2) 
% w - original opinion
% phi - fusion factor
%%%
function [w_1, w_2] = split_opinion(w, phi)
    b_1 = (w{1} * phi) / (phi * (w{1} + w{2}) + w{3});
    d_1 = (w{2} * phi) / (phi * (w{1} + w{2}) + w{3});
    u_1 = w{3} / (phi * (w{1} + w{2}) + w{3});
    a_1 = w{4};

    b_2 = ((1 - phi) * w{1}) / ((1 - phi) * (w{1} + w{2}) + w{3});
    d_2 = ((1 - phi) * w{2}) / ((1 - phi) * (w{1} + w{2}) + w{3});
    u_2 = w{3} / ((1 - phi) * (w{1} + w{2}) + w{3});
    a_2 = w{4};

    w_1 = {b_1, d_1, u_1, a_1};
    w_2 = {b_2, d_2, u_2, a_2};
end

%%%
% Split beta parameters
% beta_param_x = {r_x, s_x, a_x}
%%%
function [beta_param_1, beta_param_2] = split_beta_params(w, phi)
    r_1 = phi * 2 * w{1} / w{3};
    s_1 = phi * 2 * w{2} / w{3};
    a_1 = w{4};

    r_2 = (1 - phi) * 2 * w{1} / w{3};
    s_2 = (1 - phi) * 2 * w{2} / w{3};
    a_2 = w{4};

    beta_param_1 = {r_1, s_1, a_1};
    beta_param_2 = {r_2, s_2, a_2};
end

%%%
% Uncertanty function for opinion splitting across a path
% U - cell of uncertainty values
%%%
function U = uncertainty_in_path(w_a_b1, w_a_c1, ...
                         w_a_b2, w_c2_d, ...
                         w_b1_d, w_c1_d, ...
                         w_b2_c2, w_b2_d)
    u_ab1_d = w_a_b1{2} + w_a_b1{3} + w_a_b1{1} * w_b1_d{3};
    u_ac1_d = w_a_c1{2} + w_a_c1{3} + w_a_c1{1} * w_c1_d{3};
    u_ab2_c2_d = w_a_b2{1} * w_b2_c2{2} + w_a_b2{2} + w_a_b2{3} + ...
               w_a_b2{1} * w_b2_d{3} + w_a_b2{1} * w_b2_c2{1} * w_c2_d{3};
    
    U = {u_ab1_d, u_ac1_d, u_ab2_c2_d};
end

%%%
% Uncertanty function 
% U - cell of uncertainty values
%%%
function u_a_d = uncertainty(u_ab1_d, u_ac1_d, u_ab2_c2_d)
    u_a_d = (u_ab1_d * u_ac1_d * u_ab2_c2_d) / ...
        (u_ab1_d * u_ac1_d + u_ab1_d * u_ab2_c2_d + u_ac1_d * u_ab2_c2_d - ...
        2 * u_ab1_d * u_ac1_d * u_ab2_c2_d);
end

%%%
% Aging function
% Uses the longevity factor delta to control and forget old ratings
% t - current time
% t_r - time of the recording
%%%
function p_a_b_t = aging_rating(p_a_b_tr, delta, t, t_r)
    p_a_b_t = delta^(t - t_r) * p_a_b_tr;
end

%%%
% Aggregation function
% Aggregates ratings of a for b over time
% TODO: change to cummulative sum?
%%%
function p_t = aggregate_ratings(p_a_bs)
    p_t = sum(p_a_bs);
end

%%%
% Overall rating aggregation function
% Aggregates all ratings of b for each agenta x in S
%%%
function p_t_b = aggreagate_ratings_overall(S)
    p_t_b = sum(S);
end

%%%
% Reputation score
%%%
function rep_a_t = reputation(r, s, a)
    rep_a_t = (r + 2 * a) / (r + s + 2);
end
