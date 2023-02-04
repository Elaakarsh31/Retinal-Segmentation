im = im.to(device)
# pred_y = model(im)
# pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
# print(pred_y.shape)
# pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
# print(pred_y.shape)
# pred_y = pred_y > 0.5
# pred_y = np.array(pred_y, dtype=np.uint8)