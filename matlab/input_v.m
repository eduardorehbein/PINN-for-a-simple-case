function v = input_v(t)
    v = 10 + 2*heaviside(t-7) - 5*heaviside(t-14) + heaviside(t-21) - ...
        4*heaviside(t-28) + 4*heaviside(t-35) + 3*heaviside(t-42) + ...
        2*heaviside(t-49) - 2*heaviside(t-56) - 12*heaviside(t-63) - ...
        5*heaviside(t-70) + 3*heaviside(t-77) + 5*heaviside(t-84);
end

