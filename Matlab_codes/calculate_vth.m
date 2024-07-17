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
    end
end 