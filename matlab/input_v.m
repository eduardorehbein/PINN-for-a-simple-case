function v = input_v(t)
    if t <= 30
        v = 10 + 2*heaviside(t-6) - 5*heaviside(t-11) + heaviside(t-18) - 4*heaviside(t-24);
    elseif t <= 60
        v = 3 + 4*heaviside(t-36) - 2*heaviside(t-41) + 2*heaviside(t-48) - 6*heaviside(t-54);
    else
        v = -5 + 12*heaviside(t-63) - 5*heaviside(t-70) + 3*heaviside(t-74) + 6*heaviside(t-83);
    end
end

