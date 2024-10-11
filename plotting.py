def plot_res(model, x,y,z, u_exact, v_exact , w_exact, p_exact):
    # Convert tensors to numpy arrays
    xx = x[:50]
    yy = y[:50]
    zz = z[:50]
    uu = u_exact[:50]
    vv = v_exact[:50]
    ww = w_exact[:50]
    pp = p_exact[:50]
    
    x_train_np = xx.cpu().detach().numpy()
    y_train_np = yy.cpu().detach().numpy()
    z_train_np = zz.cpu().detach().numpy()
    u_exact_np = uu.cpu().detach().numpy()
    v_exact_np = vv.cpu().detach().numpy()
    w_exact_np = ww.cpu().detach().numpy()
    p_exact_np = pp.cpu().detach().numpy()
    
    uvp_pred = model(torch.cat((xx, yy, zz), dim=1))
    u_pred = uvp_pred[:, 0:1]
    v_pred = uvp_pred[:, 1:2]
    w_pred = uvp_pred[:, 2:3]
    p_pred = uvp_pred[:, 3:4]
    
    u_pred_np = u_pred.cpu().detach().numpy()
    v_pred_np = v_pred.cpu().detach().numpy()
    w_pred_np = w_pred.cpu().detach().numpy()
    p_pred_np = p_pred.cpu().detach().numpy()
    
    plt.figure(dpi = 150)
    plt.plot(u_exact_np, label='Exact u', marker='o')
    plt.plot(u_pred_np, label='Predicted u', marker='x')
    plt.legend()
    plt.title('Comparison of u component')