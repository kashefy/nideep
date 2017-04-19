'''
Created on Apr 19, 2017

@author: kashefy
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.calibration import calibration_curve
import nideep.eval.log_utils as lu
from nideep.eval.learning_curve import LearningCurve

colors_base = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.3', '0.5', '0.75', 'chartreuse', 'burlywood', 'aqua']

def find_nearest(arr, value):
    idx = np.abs(arr-value).argmin()
    return idx

def eval_id_roc(gt, preds, classnames=None,
                thr_select=[0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.999],
                do_plot=True):
    """
    Evaluation metric Identification: ROC (class-absent-bin), with threshold markers
    """
    num_classes = gt.shape[-1]
    if classnames is None:
        classnames = range(num_classes)
    fig, ax = plt.subplots(1, 1)
    perf = []
    for cl in xrange(num_classes):
        gt_cl = gt[:, cl].ravel()
        preds_cl = preds[:, cl].ravel()
        fpr, tpr, thr = metrics.roc_curve(gt_cl, preds_cl,
                                          drop_intermediate=True)
        roc_auc = metrics.auc(fpr, tpr)
        if do_plot:
            ax.plot(fpr, tpr,
                    color=colors_base[cl],
                    label='%d. %s ROC (auc=%2f)' % (cl, classnames[cl], roc_auc,))
            if len(thr_select) > 0:
                fig_cl, ax_cl = plt.subplots(1, 1)
                ax_cl.plot(fpr, tpr,
                         color=colors_base[cl],
                         label='%d. %s ROC (auc=%2f)' % (cl, classnames[cl], roc_auc,))
                thr = np.array(thr)
                for thr_s in thr_select:
                    thr_idx = find_nearest(thr, thr_s)
                    ax_cl.scatter(fpr[thr_idx], tpr[thr_idx],
                               color=colors_base[cl])
                    ax_cl.text(fpr[thr_idx]+0.01, tpr[thr_idx]-0.02,
                               '%f' % (thr[thr_idx],),
                               color=colors_base[cl],
                               size=10)
                ax_cl.set_xlabel('false positive rate (fall-out)')
                ax_cl.set_ylabel('true positive rate (sensitivity)')
                ax_cl.set_title("%s ROC" % (classnames[cl],))
                ax_cl.set_xlim([-0.02,1.02])
                ax_cl.set_ylim([-0.02,1.02])
                ax_cl.legend(loc='lower right')
        perf.append({'cl_id': cl, 'classname': classnames[cl],
                     'roc_auc': roc_auc})
    ax.set_xlabel('false positive rate (fall-out)')
    ax.set_ylabel('true positive rate (sensitivity)')
    ax.set_title("ROC")
    ax.set_xlim([-0.02,1.02])
    ax.set_ylim([-0.02,1.02])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    return perf

def eval_id_pr(gt, preds, classnames=None,
               thr_select=[0.001, 0.01, 0.1, 0.5, 0.9, 0.999]):
    """
    Evaluation metric Identification: Precision-Recall curve (class-absent-bin)
    """
    num_classes = gt.shape[-1]
    if classnames is None:
        classnames = range(num_classes)
    fig, ax = plt.subplots(1, 1)
    perf = []
    for cl in xrange(num_classes):
        gt_cl = gt[:, cl].ravel()
        preds_cl = preds[:, cl].ravel()
        precision, recall, thr = metrics.precision_recall_curve(gt_cl, preds_cl)
        pr_auc = metrics.auc(recall, precision)
        ax.plot(recall, precision,
                color=colors_base[cl],
                label='%d. %s (auc=%2f)' % (cl, classnames[cl],
                                            pr_auc,))
        if len(thr_select) > 0:
            fig_cl, ax_cl = plt.subplots(1, 1)
            ax_cl.plot(recall, precision,
                       color=colors_base[cl],
                       label='%d. %s (auc=%2f)' % (cl, classnames[cl],
                                                   pr_auc,))
            thr = np.array(thr)
            for thr_s in thr_select:
                thr_idx = find_nearest(thr, thr_s)
                ax_cl.scatter(recall[thr_idx], precision[thr_idx],
                            color=colors_base[cl])
                ax_cl.text(recall[thr_idx]*1.008, precision[thr_idx]*1.002,
                         '%f' % (thr[thr_idx],),
                         color=colors_base[cl],
                         size=10)
                ax_cl.set_xlabel('recall (true positive rate, sensitivity)')
                ax_cl.set_ylabel('precision %s' % (r'$TP/(TP+FP)$',))
                ax_cl.set_title("%s Precision-Recall" % (classnames[cl],))
                ax_cl.set_xlim([-0.02,1.02])
                ax_cl.set_ylim([-0.02,1.02])
                ax_cl.legend(loc='lower right')
        perf.append({'cl_id': cl, 'classname': classnames[cl],
                     'pr_auc': pr_auc,
                     'num_points': len(gt_cl),
                     'num_pos': np.count_nonzero(gt_cl),
                     })
    ax.set_xlabel('recall (true positive rate, sensitivity)')
    ax.set_ylabel('precision %s' % (r'$TP/(TP+FP)$',))
    ax.set_title("per-class Precision-Recall")
    ax.set_xlim([-0.02,1.02])
    ax.set_ylim([-0.02,1.02])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    return perf

def eval_id_bacc(gt, preds, classnames=None):
    """ Identification: Balanced Accuracy """
    num_classes = gt.shape[-1]
    if classnames is None:
        classnames = range(num_classes)
    fig, ax2 = plt.subplots(1, 1)
    perf = []
    for cl in xrange(num_classes):
        # sensitivity == true positive rate (TPR) == hit rate == recall
        # specificity (SPC) or true negative rate (TNR) TN/N, equivalent to 1 - FPR
        gt_cl = gt[:, cl].ravel()
        preds_cl = preds[:, cl].ravel()
        fpr, sensitivity, thr = metrics.roc_curve(gt_cl, preds_cl,
                                                  drop_intermediate=True)
        spc = 1-fpr
        bacc_cl = 0.5*np.array(sensitivity)+0.5*np.array(spc)
        bacc_cl_max = bacc_cl.max()
        thr_bacc_cl_max = thr[bacc_cl.argmax()]
        ax2.plot(thr, bacc_cl,
                 color=colors_base[cl],
                 label=
                 '%d. %s, max(BAcc)=%.3f at thr=%.7f' % (cl,
                                                         classnames[cl],
                                                         bacc_cl_max,
                                                         thr_bacc_cl_max,
                                                        ))
        perf.append({'cl_id': cl, 'classname': classnames[cl],
                     'bacc_cl_max': bacc_cl_max,
                     'sensitivity_bacc_cl_max': sensitivity[bacc_cl.argmax()],
                     'specificity_bacc_cl_max': spc[bacc_cl.argmax()],
                     'thr_bacc_cl_max': thr_bacc_cl_max,
                     'num_points': len(gt_cl),
                     'num_pos': np.count_nonzero(gt_cl),
                     }) 
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_title('Per-class Balanced Accuracy')
    ax2.set_xlabel('classifier threshold')
    ax2.set_ylabel('Balanced Accuracy')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    return perf

def eval_id_activation(gt, preds, classnames=None):
    """ Identification: activation distribution
    """
    num_classes = gt.shape[-1]
    if classnames is None:
        classnames = range(num_classes)
    #fig, arr_ax = plt.subplots(nrows=num_classes+1, ncols=1)
    num_bins = 30
    fig_cal, ax_cal = plt.subplots(1, 1)
    for cl in xrange(num_classes):
        gt_cl = gt[:, cl].ravel()
        preds_cl = preds[:, cl].ravel()
        frac_positives, mean_pred = \
        calibration_curve(gt_cl, preds_cl,
                          normalize=False, n_bins=num_bins)
        
        fig_act, ax_act = plt.subplots(1, 1)
        ax_cal.plot(mean_pred, frac_positives,
                        color=colors_base[cl],
                         label='%d. %s' % (cl, classnames[cl],))
        gt_cl_pos = gt_cl==1
        ax_act.hist(preds_cl[gt_cl_pos],
                    bins=num_bins, color='b')
        ax_act.hist(preds_cl[np.logical_not(gt_cl_pos)],
                    bins=num_bins, color='r', alpha=0.4)
        ax_act.set_yscale('log', nonposy='clip')
        ax_act.set_xlabel('activation')
        ax_act.set_ylabel('count')
        ax_act.set_title("%s activation histogram" % (classnames[cl],))
    ax_cal.set_xlabel('mean activation')
    ax_cal.set_ylabel('fraction of positive samples')
    ax_cal.set_title('calibration reliability')
    ax_cal.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    
def eval_id_ge_thr(gt, preds, thr, classnames=None):
    """ Identification: Performance at thr
    """
    num_classes = gt.shape[-1]
    if classnames is None:
        classnames = range(num_classes)
    perf = []
    for cl in xrange(num_classes):
        gt_cl = gt[:, cl].ravel() == 1
        preds_cl = preds[:, cl].ravel() >= thr[cl]
        tp = np.sum(np.logical_and(preds_cl, gt_cl))
        tn = np.sum(np.logical_and(np.logical_not(preds_cl), np.logical_not(gt_cl)))
        fp = np.sum(np.logical_and(preds_cl, np.logical_not(gt_cl)))
        fn = np.sum(np.logical_and(np.logical_not(preds_cl), gt_cl))
        p = float(np.count_nonzero(gt_cl))
        n = float(gt_cl.size - p)
        sensitivity = tp/p
        specificity = tn/n
        bacc = 0.5*sensitivity+0.5*specificity
        precision = tp / float(tp + fp)
        f1 = 2 * (precision * sensitivity) / float(precision + sensitivity)
        perf.append({'cl_id': cl, 'classname': classnames[cl],
                     'bacc': bacc,
                     'sensitivity': sensitivity,
                     'specificity': specificity,
                     'precision': precision,
                     'f1': f1,
                     'thr': thr[cl],
                     'num_points': len(gt_cl),
                     'num_pos': p,
                     })
    return perf

def eval_loc_ge_thr(gt, preds, thr):
    """ Localization: Performance at thr """
    locs = gt.shape[-1]
    loc_azimuth_value = np.linspace(-180, 180, locs, endpoint=False)
    perf = []
    for loc_idx in xrange(locs):
        gt_loc = gt[:, :, :, loc_idx].ravel() == 1
        if len(preds.shape) == 3:
            pred_loc = preds[:, np.newaxis, :, loc_idx].ravel() >= thr[loc_idx]
        else:
            pred_loc = preds[:, :, :, loc_idx].ravel() >= thr[loc_idx]

        tp = np.sum(np.logical_and(pred_loc, gt_loc))
        tn = np.sum(np.logical_and(np.logical_not(pred_loc), np.logical_not(gt_loc)))
        fp = np.sum(np.logical_and(pred_loc, np.logical_not(gt_loc)))
        fn = np.sum(np.logical_and(np.logical_not(pred_loc), gt_loc))
        p = float(np.count_nonzero(gt_loc))
        n = float(gt_loc.size - p)
        sensitivity = tp/p
        specificity = tn/n
        bacc = 0.5*sensitivity+0.5*specificity
        precision = tp / float(tp + fp)
        f1 = 2 * (precision * sensitivity) / float(precision + sensitivity)
        perf.append({'cl_id': loc_idx, 'classname': r'%s$^\circ$' % (loc_azimuth_value[loc_idx],),
                     'bacc': bacc,
                     'sensitivity': sensitivity,
                     'specificity': specificity,
                     'precision': precision,
                     'f1': f1,
                     'thr': thr[loc_idx],
                     'num_points': len(gt_loc),
                     'num_pos': p,
                     })
    return perf

def eval_loc_pr(gt, preds):
    """ Localization: Azimuth Precision-Recall curve + AUC(PR) polar plot"""   
    locs = gt.shape[-1]
    loc_azimuth_value = np.linspace(-180, 180, locs, endpoint=False)
    pr_auc_vals = np.zeros(loc_azimuth_value.shape)
    plt.figure()
    perf = []
    for loc_idx in xrange(locs):
        gt_loc = gt[:, :, :, loc_idx].ravel()
        if len(preds.shape) == 3:
            pred_loc = preds[:, np.newaxis, :, loc_idx].ravel()
        else:
            pred_loc = preds[:, :, :, loc_idx].ravel()
        precision, recall, _ = metrics.precision_recall_curve(gt_loc, pred_loc)
        pr_auc = metrics.auc(recall, precision)
        plt.plot(recall, precision,
                 'b-',
                 label='%d. (auc=%2f)' % (loc_azimuth_value[loc_idx], pr_auc,))
        pr_auc_vals[loc_idx] = pr_auc
        perf.append({'cl_id': loc_idx, 'classname': r'%s$^\circ$' % (loc_azimuth_value[loc_idx],),
                     'pr_auc': pr_auc,
                     'num_points': len(gt_loc),
                     'num_pos': np.count_nonzero(gt_loc),
                     })
    plt.xlabel('recall (true positive rate, sensitivity)')
    plt.ylabel('precision %s' % (r'$TP/(TP+FP)$',))
    plt.title("per-azimuth Precision-Recall")
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    # repeat first value to complete the circle
    loc_azimuth_value = np.append(loc_azimuth_value, loc_azimuth_value[0])
    pr_auc_vals = np.append(pr_auc_vals, pr_auc_vals[0])
    ax.plot(np.deg2rad(loc_azimuth_value), pr_auc_vals, color='b', linewidth=2.)
    #ax.plot(np.deg2rad([45,90,180,-180,-45]), [0.1,0.2,0.3,0.4,0.5], color='r', linewidth=3.)
    ax.set_rmax(1.0)
    ax.set_xticklabels([r'%s$^\circ$' % (tmp,) for tmp in ['0', '45', '90', '135', '180', '-135', '-90', '-45']])
    ax.set_theta_zero_location('N')  # Set zero to North
    ax.set_theta_direction(-1)  # -1: Theta increases in the clockwise direction
    ax.grid(b=True, which='both')
    ax.minorticks_on()
    ax.set_title('Per-azimuth auc(Precision-Recall)')
    head_img = read_png('/home/kashefy/src/caffe_pvt/head.png')
    imagebox = OffsetImage(1-head_img, zoom=0.4, cmap='Greys')
    ab = AnnotationBbox(imagebox, [0, 0],
                        xycoords='data',
                        pad=-0.1)
    ax.add_artist(ab)
    return perf

def eval_loc_bacc(gt, preds):
    """Localization: Azimuth Balanced Accuracy + max(BAcc) polar plot"""
    locs = gt.shape[-1]
    loc_azimuth_value = np.linspace(-180, 180, locs, endpoint=False)
    radii = np.zeros(loc_azimuth_value.shape)
    perf = []
    for loc_idx in xrange(locs):
        gt_loc = gt[:, :, :, loc_idx].ravel()
        if len(preds.shape) == 3:
            pred_loc = preds[:, np.newaxis, :, loc_idx].ravel()
        else:
            pred_loc = preds[:, :, :, loc_idx].ravel()
        pred_loc.shape
        fpr, sensitivity, thr = metrics.roc_curve(gt_loc, pred_loc,
                                                  drop_intermediate=True)
        spc = 1-fpr
        bacc_cl = 0.5*np.array(sensitivity)+0.5*np.array(spc)
        radii[loc_idx] = bacc_cl.max()
        perf.append({'cl_id': loc_idx, 'classname': r'%s$^\circ$' % (loc_azimuth_value[loc_idx],),
                     'bacc_cl_max': radii[loc_idx],
                     'sensitivity_bacc_cl_max': sensitivity[bacc_cl.argmax()],
                     'specificity_bacc_cl_max': spc[bacc_cl.argmax()],
                     'thr_bacc_cl_max': thr[bacc_cl.argmax()],
                     'num_points': len(gt_loc),
                     'num_pos': np.count_nonzero(gt_loc),
                     }) 
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    # repeat first value to complete the circle
    loc_azimuth_value = np.append(loc_azimuth_value, loc_azimuth_value[0])
    radii = np.append(radii, radii[0])
    ax.plot(np.deg2rad(loc_azimuth_value), radii, color='b', linewidth=2.)
    ax.set_rmax(1.0)
    ax.set_xticklabels([r'%s$^\circ$' % (tmp,) for tmp in ['0', '45', '90', '135', '180', '-135', '-90', '-45']])
    ax.set_theta_zero_location('N')  # Set zero to North
    ax.set_theta_direction(-1)  # -1: Theta increases in the clockwise direction
    ax.grid(b=True, which='both')
    ax.minorticks_on()
    ax.set_title('Per-azimuth max(BAcc)')
    head_img = read_png('/home/kashefy/src/caffe_pvt/head.png')
    imagebox = OffsetImage(1-head_img, zoom=0.4, cmap='Greys')
    ab = AnnotationBbox(imagebox, [0, 0],
                        xycoords='data',
                        pad=-0.1)
    ax.add_artist(ab)
    return perf

def eval_loc_bacc_cond(gt, preds, preds_id):
    """Conditioned Localization BAcc|Identification"""
    locs = gt.shape[-1]-1
    loc_azimuth_value = np.linspace(-180, 180, locs, endpoint=False)
    radii = np.zeros(loc_azimuth_value.shape)
    perf = []
    for loc_idx in xrange(locs):
        gt_loc = gt[:, :, :, loc_idx].ravel()
        if len(preds.shape) == 3:
            pred_loc = preds[:, np.newaxis, :, loc_idx].ravel()
        else:
            pred_loc = preds[:, :, :, loc_idx].ravel()
        pred_loc = pred_loc * (1-preds_id.ravel())
        fpr, sensitivity, thr = metrics.roc_curve(gt_loc, pred_loc,
                                                  drop_intermediate=True)
        spc = 1-fpr
        bacc_cl = 0.5*np.array(sensitivity)+0.5*np.array(spc)
        radii[loc_idx] = bacc_cl.max()
        perf.append({'cl_id': loc_idx, 'classname': r'%s$^\circ$' % (loc_azimuth_value[loc_idx],),
                     'bacc_cl_max': radii[loc_idx],
                     'sensitivity_bacc_cl_max': sensitivity[bacc_cl.argmax()],
                     'specificity_bacc_cl_max': spc[bacc_cl.argmax()],
                     'thr_bacc_cl_max': thr[bacc_cl.argmax()],
                     'num_points': len(gt_loc),
                     'num_pos': np.count_nonzero(gt_loc),
                     }) 
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    # repeat first value to complete the circle
    loc_azimuth_value = np.append(loc_azimuth_value, loc_azimuth_value[0])
    radii = np.append(radii, radii[0])
    ax.plot(np.deg2rad(loc_azimuth_value), radii, color='b', linewidth=2.)
    ax.set_rmax(1.0)
    ax.set_xticklabels([r'%s$^\circ$' % (tmp,) for tmp in ['0', '45', '90', '135', '180', '-135', '-90', '-45']])
    ax.set_theta_zero_location('N')  # Set zero to North
    ax.set_theta_direction(-1)  # -1: Theta increases in the clockwise direction
    ax.grid(b=True, which='both')
    ax.minorticks_on()
    ax.set_title('Per-azimuth max(BAcc|Identification)')
    head_img = read_png('/home/kashefy/src/caffe_pvt/head.png')
    imagebox = OffsetImage(1-head_img, zoom=0.4, cmap='Greys')
    ab = AnnotationBbox(imagebox, [0, 0],
                        xycoords='data',
                        pad=-0.1)
    ax.add_artist(ab)
    return perf

def imshow_gt_preds_at(gt, preds, idx=None, shape=None):
    if idx is None:
        idx = np.random.randint(0, high=len(gt))
    plt.figure()
    plt.subplot(211)
    plt.imshow(np.squeeze(gt[idx]), interpolation='none')
    plt.subplot(212)
    if shape is None:
        plt.imshow(np.squeeze(preds[idx], interpolation='none'))
    else:
        plt.imshow(np.squeeze(preds[idx].reshape(shape[0],shape[1])), interpolation='none')
    plt.colorbar()
    
def plot_gt_preds_at(gt, preds, idx=None, shape=None):
    """Show ground truth + predictions at random index"""
    if idx is None:
        idx = np.random.randint(0, high=len(gt))
    #fig = plt.figure(figsize=(10,5))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    locs = gt.shape[-1]-1
    gt_i = np.squeeze(gt[idx])[:,:-1]
    if shape is None:
        preds_i = np.squeeze(preds[idx])
    else:
        preds_i = np.squeeze(preds[idx].reshape(shape[0],shape[1]))
    if preds_i.shape[-1] == gt_i.shape[-1]+1:
        preds_i = preds_i[:,:-1]
#     print gt_i.shape
    # repeat first value to complete the circle
    loc_azimuth_value = np.linspace(-180, 180, locs, endpoint=False)
#     print loc_azimuth_value.shape, preds_i[:,0].reshape(gt_i.shape[0],1).shape
    loc_azimuth_value = np.append(loc_azimuth_value, loc_azimuth_value[0])
    radii = np.concatenate((preds_i, preds_i[:,0].reshape(gt_i.shape[0],1)), 1)
    for cl_idx in xrange(len(classnames)):
        #print 'x', cl_idx, radii.shape, radii[cl_idx,:].shape, loc_azimuth_value.shape
        ax.plot(np.deg2rad(loc_azimuth_value), radii[cl_idx, :]*1,
                color=colors_base[cl_idx], linewidth=2., label=classnames[cl_idx],
               zorder=10)
        for loc_idx in xrange(locs):
            if gt_i[cl_idx, loc_idx] > 0:
                ax.plot(np.deg2rad(loc_azimuth_value[loc_idx]), gt_i[cl_idx, loc_idx],
                        color=colors_base[cl_idx], linewidth=9.,
                        marker='*', markersize=15, zorder=11)
    ax.set_rmax(1.1)
    ax.set_xticklabels([r'%s$^\circ$' % (tmp,) for tmp in ['0', '45', '90', '135', '180', '-135', '-90', '-45']])
    ax.set_theta_zero_location('N')  # Set zero to North
    ax.set_theta_direction(-1)  # -1: Theta increases in the clockwise direction
    ax.grid(b=True, which='both')
    #ax.minorticks_on()
    fig.suptitle('Joint sound identification and localization', y=1.02)
    head_img = read_png('/home/kashefy/src/caffe_pvt/head.png')
    imagebox = OffsetImage(1-head_img, zoom=0.4, cmap='Greys', zorder=1)
    ab = AnnotationBbox(imagebox, [0, 0],
                        xycoords='data',
                        pad=-0.1,
                       )
    ax.add_artist(ab)
    ax.set_axisbelow(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=1)
    
def learning_curves(logs, ylim=None, xlim=None):
    """ Network Learning Curves """
    for phase in [Phase.TRAIN, Phase.TEST]:
        print phase
        fig, ax = plt.subplots(1, 1)
        for p in logs:
            e = LearningCurve(p)
            lc_keys = e.parse()[phase == Phase.TEST]
            num_iter = e.list('NumIters', phase)
            print('%s %s: %d %s iterations' % (os.path.basename(os.path.dirname(p)), lu.pid_from_logname(p), num_iter.size, phase))
            for lck_idx, lck in enumerate(lc_keys):
                if 'nidx' in lck or ('NumIters' not in lck and 'rate' not in lck.lower() and 'seconds' not in lck.lower()):
                    try:
                        loss = e.list(lck, phase)
                        ax.plot(num_iter, loss, label='%s %s %d' % (os.path.basename(os.path.dirname(p)), lck, lu.pid_from_logname(p)))
                    except KeyError as kerr:
                        print("Inavlid values for %s %s" % (phase, lck))
        ticks = ax.get_xticks()
        ax.set_xticklabels(["%dK" % int(t/1000) for t in ticks])

        ax.set_xlabel('iterations')
        ax.set_ylabel(' '.join([phase, 'cross entropy loss']))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title('on %s set' % phase)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        ax.grid()
        
def count_pos_examples():
    print bacc[0].keys()
    n = len(h['label_id_loc'])
    print('n=%d' % (n,))
    gt = 1-h['label_id_loc'][:1000000]
    for bacc_cl_info in bacc:
        print bacc_cl_info['cl_id'], bacc_cl_info['classname']
        pos = gt[:, :, bacc_cl_info['cl_id'], -1]
        num_pos = np.count_nonzero(pos)
        print num_pos, num_pos/float(n)*100
        
if __name__ == '__main__':
    pass