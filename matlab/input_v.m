function v = input_v(t)
    v = 10*heaviside(t) + 2*heaviside(t-6) - 5*heaviside(t-12) + heaviside(t-18) - 4*heaviside(t-24);
end

