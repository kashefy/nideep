'''
Created on Apr 19, 2017

@author: kashefy
'''
import pickle
import numpy as np
import h5py
from nideep.datasets.twoears.teval import * # TODO: FIX

def save_obj(fpath, obj):
    with open(fpath + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fpath):
    with open(fpath + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_info(fpath):
    with open(fpath + '.info', 'rb') as f:
        lines = f.readlines()
        lines = [l.rstrip('\n') for l in lines]
        lines = [l.rstrip(',MLP') for l in lines]
        lines = [l.replace('MLP+', '') for l in lines]
        return lines

def get_model_keys(h, key_label='label_id_loc'):

    model_keys = []
    for key in h.keys():
#         print('%s: %s' % (key, h[key].shape))
        if 'void' in key:
            model_keys.append({'pred_name' : key,
                               'pred_type' : 'id',
                               'do_invert' : True})
        elif 'bin' in key:
            model_keys.append({'pred_name' : key,
                               'pred_type' : 'loc',
                               'do_invert' : False})
        elif key != key_label and 'srcs' not in key.lower():
            model_keys.append({'pred_name' : key,
                               'pred_type' : 'id',
                               'do_invert' : True})
            model_keys.append({'pred_name' : key,
                               'pred_type' : 'loc',
                               'do_invert' : False})
    return model_keys
    
def eval_summary(fpath_infer, num_points=None, key_label='label_id_loc'):
    
    with h5py.File(fpath_infer, 'r') as h:
        if num_points is None:
            num_points = len(h[key_label])
        model_stats = get_model_keys(h, key_label=key_label)

        for m in model_stats:
            key_pred = m['pred_name']
            gt = h[key_label]
            pred = h[key_pred]
    #         print h.file, key_pred, pred.shape, m['pred_type']
            if m['pred_type'] == 'id':
                gt = np.squeeze(gt[:num_points,:,:,-1])
                # handle different prediction shapes
                if len(pred.shape) == 2:
                    if pred.shape[1] > len(classnames):
                        pred = pred[:num_points].reshape(num_points, 1, len(classnames), num_azimuths+1)
                    else:
                        pred = pred[:num_points].reshape(num_points, 1, len(classnames), 1)
                if len(pred.shape) == 4:
                    pred = np.squeeze(pred[:num_points,:,:,-1])
                if m['do_invert']:
                    gt = 1-gt
                    pred = 1-pred
                m['id_roc'] = eval_id_roc(gt, pred, classnames=classnames)
                m['id_bacc'] = eval_id_bacc(gt, pred, classnames=classnames)
                m['id_pr'] = eval_id_pr(gt, pred, classnames=classnames)
            elif m['pred_type'] == 'loc':
                gt = gt[:num_points,:,:,:-1] # no squeeze
                # handle different prediction shapes
                if len(pred.shape) == 2 and pred.shape[1] > num_azimuths:
                        pred = pred[:num_points].reshape(num_points, 1, len(classnames), num_azimuths+1)
                if len(pred.shape) == 4:
                    # exclude void bin if present
                    if pred.shape[-1] > num_azimuths:
                        pred_id = pred[:num_points,:,:,-1]
                        pred = pred[:num_points,:,:,:-1]
                        m['loc_bacc_cond'] = eval_loc_bacc_cond(gt, pred, pred_id)
                    else:
                        pred = pred[:num_points,:,:,:]
    #             print gt.shape, pred.shape
                m['loc_pr'] = eval_loc_pr(gt, pred)
                m['loc_bacc'] = eval_loc_bacc(gt, pred)
    return model_stats

def eval_summary_cond_nSrcs(fpath_infer, num_points=None,
                       key_label='label_id_loc',
                        key_label_nSRc='label_nSrcs',
                           ):
    """Evaluation summary given no. of sources"""
    with h5py.File(fpath_infer, 'r') as h:
        if num_points is None:
            num_points = len(h[key_label])
        nSrcs = h[key_label_nSRc][:num_points].flatten()
        
        model_stats_nSrcs = []
        
        for nSrcs_val in xrange(5):
            print nSrcs.shape
            model_stats = get_model_keys(h, key_label=key_label)
            
            print 'nSrcs_val: ', nSrcs_val
            nSrcs_cur = np.where(nSrcs==nSrcs_val)[0]
            num_points_orig = num_points
            for m in model_stats:
                num_points = num_points_orig
                key_pred = m['pred_name']
#                 print 'pred', key_pred, m['pred_type']
                gt = h[key_label][:num_points][nSrcs_cur]
                pred = h[key_pred][:num_points][nSrcs_cur]
                if m['pred_type'] == 'id':
                    # handle different prediction shapes
                    gt = np.squeeze(gt[:,:,:,-1])
                    if len(pred.shape) == 2:
                        if pred.shape[1] > len(classnames):
                            pred = pred.reshape(len(nSrcs_cur), 1, len(classnames), num_azimuths+1)
                        else:
                            pred = pred.reshape(len(nSrcs_cur), 1, len(classnames), 1)
                    if len(pred.shape) == 4:
                        pred = np.squeeze(pred[:,:,:,-1])
                    if m['do_invert']:
                        gt = 1-gt
                        pred = 1-pred
#                     m['id_roc'] = eval_id_roc(gt, pred, classnames=classnames)
#                     print 'minmax', gt.min(), gt.max()
                    m['id_bacc'] = eval_id_bacc(gt, pred, classnames=classnames)
#                     m['id_pr'] = eval_id_pr(gt, pred, classnames=classnames)
                elif m['pred_type'] == 'loc':
                    gt = gt[:,:,:,:-1] # no squeeze
                    # handle different prediction shapes
                    if len(pred.shape) == 2 and pred.shape[1] > num_azimuths:
                            pred = pred.reshape(num_points, 1, len(classnames), num_azimuths+1)
                    if len(pred.shape) == 4:
                        # exclude void bin if present
                        if pred.shape[-1] > num_azimuths:
                            pred = pred[:,:,:,:-1]
                        else:
                            pred = pred[:,:,:,:]
#                     m['loc_pr'] = eval_loc_pr(gt, pred)
                    m['loc_bacc'] = eval_loc_bacc(gt, pred)
            model_stats_nSrcs.append(model_stats)
            print 'model_stats_nSrcs', len(model_stats_nSrcs)
    return model_stats_nSrcs

def eval_summary_cond_nSrcs_thr(fpath_infer, num_points=None,
                                key_label='label_id_loc',
                                key_label_nSRc='label_nSrcs',
                           ):
    """Thr. Evaluation summary given no. of sources"""
    model_stats_overall = load_obj(fpath_infer)
    with h5py.File(fpath_infer, 'r') as h:
        if num_points is None:
            num_points = len(h[key_label])
        nSrcs = h[key_label_nSRc][:num_points].flatten()
        
        model_stats_nSrcs = []
        
        for nSrcs_val in xrange(5):
#             print nSrcs.shape
            model_stats = get_model_keys(h, key_label=key_label)
            
#             print 'nSrcs_val: ', nSrcs_val
            nSrcs_cur = np.where(nSrcs==nSrcs_val)[0]
            num_points_orig = num_points
            for m, movrl in zip(model_stats, model_stats_overall):
                num_points = num_points_orig
                key_pred = m['pred_name']
#                 print 'pred', key_pred, m['pred_type']
                gt = h[key_label][:num_points][nSrcs_cur]
                pred = h[key_pred][:num_points][nSrcs_cur]
                if m['pred_type'] == 'id':
                    # handle different prediction shapes
                    gt = np.squeeze(gt[:,:,:,-1])
                    if len(pred.shape) == 2:
                        if pred.shape[1] > len(classnames):
                            pred = pred.reshape(len(nSrcs_cur), 1, len(classnames), num_azimuths+1)
                        else:
                            pred = pred.reshape(len(nSrcs_cur), 1, len(classnames), 1)
                    if len(pred.shape) == 4:
                        pred = np.squeeze(pred[:,:,:,-1])
                    if m['do_invert']:
                        gt = 1-gt
                        pred = 1-pred
#                     m['id_roc'] = eval_id_roc(gt, pred, classnames=classnames)
#                     print 'minmax', gt.min(), gt.max()
#                     print pred[:10]
                    thr = [c['thr_bacc_cl_max'] for c in movrl['id_bacc']]
                    m['id_ge_thr'] = eval_id_ge_thr(gt, pred, thr, classnames=classnames)
                elif m['pred_type'] == 'loc':
                    gt = gt[:,:,:,:-1] # no squeeze
                    # handle different prediction shapes
                    if len(pred.shape) == 2 and pred.shape[1] > num_azimuths:
                            pred = pred.reshape(num_points, 1, len(classnames), num_azimuths+1)
                    if len(pred.shape) == 4:
                        # exclude void bin if present
                        if pred.shape[-1] > num_azimuths:
                            pred = pred[:,:,:,:-1]
                        else:
                            pred = pred[:,:,:,:]
                    thr = [c['thr_bacc_cl_max'] for c in movrl['loc_bacc']]
                    m['loc_ge_thr'] = eval_loc_ge_thr(gt, pred, thr)
            model_stats_nSrcs.append(model_stats)
            print 'model_stats_nSrcs', len(model_stats_nSrcs)
    return model_stats_nSrcs

def eval_summary_mixed(fpath_infer, num_points=None,
                       key_label='label_id_loc'):
    """ Evaluation summary given mixtures with other classes """
    h = h5py.File(fpath_infer, 'r')
    if num_points is None:
        num_points = len(h[key_label])
    model_stats = get_model_keys(h, key_label=key_label)
    num_points_orig = num_points
    for m in model_stats:
        num_points = num_points_orig
        key_pred = m['pred_name']
        gt = h[key_label][:num_points,:,:,:-1]
        pred = h[key_pred][:num_points]
        source_count = np.sum(gt, axis=(1,2,3))
        mix_rows = np.where(source_count>1)[0]
        num_points = min(mix_rows.size, num_points)
        mix_rows = mix_rows[:num_points]
        gt = gt[mix_rows]
        pred = pred[mix_rows]
        if m['pred_type'] == 'id':
            # handle different prediction shapes
            gt = np.squeeze(gt[:num_points,:,:,-1])
            if len(pred.shape) == 2:
                if pred.shape[1] > len(classnames):
                    pred = pred[:num_points].reshape(num_points, 1, len(classnames), num_azimuths+1)
                else:
                    pred = pred[:num_points].reshape(num_points, 1, len(classnames), 1)
            if len(pred.shape) == 4:
                pred = np.squeeze(pred[:num_points,:,:,-1])
            if m['do_invert']:
                gt = 1-gt
                pred = 1-pred
            m['id_roc'] = eval_id_roc(gt, pred, classnames=classnames)
            m['id_bacc'] = eval_id_bacc(gt, pred, classnames=classnames)
            m['id_pr'] = eval_id_pr(gt, pred, classnames=classnames)
        elif m['pred_type'] == 'loc':
            gt = gt[:num_points,:,:,:-1] # no squeeze
            # handle different prediction shapes
            if len(pred.shape) == 2 and pred.shape[1] > num_azimuths:
                    pred = pred[:num_points].reshape(num_points, 1, len(classnames), num_azimuths+1)
            if len(pred.shape) == 4:
                # exclude void bin if present
                if pred.shape[-1] > num_azimuths:
                    pred = pred[:num_points,:,:,:-1]
                else:
                    pred = pred[:num_points,:,:,:]
            m['loc_pr'] = eval_loc_pr(gt, pred)
            m['loc_bacc'] = eval_loc_bacc(gt, pred)     
    return model_stats

def eval_summary_cond100(fpath_infer, num_points=None,
                       key_label='label_id_loc',
                        key_label_nSRc='label_nSrcs',
                           ):
    """100 condition set"""
    model_stats_overall = load_obj(fpath_infer)
    with open('/mnt/raid/data/ni/twoears/kashefy/brir/saveData3_id_loc_nSrcs_test/twoears_data_test.txt', 'r') as f:
        paths_conds = f.readlines()
    paths_conds = [x.replace('\n', '') for x in paths_conds]
#     paths_conds = paths_conds[:10]

    with h5py.File(fpath_infer, 'r') as h:
        if num_points is None:
            num_points = len(h[key_label])
        nSrcs = h[key_label_nSRc][:num_points].flatten()
        
        model_stats_cond = []
        start_idx = 0
        for cond_idx, path_cond in enumerate(paths_conds):
            with h5py.File(path_cond, 'r') as h_cond:
                model_stats = get_model_keys(h, key_label=key_label)

                print 'cond_idx: ', cond_idx
                end_idx = start_idx + len(h_cond[key_label])
                for m, movrl in zip(model_stats, model_stats_overall):
                    key_pred = m['pred_name']
    #                 print 'pred', key_pred, m['pred_type']
                    gt = h[key_label][start_idx:end_idx]
                    pred = h[key_pred][start_idx:end_idx]
                    if m['pred_type'] == 'id':
                        # handle different prediction shapes
                        gt = np.squeeze(gt[:,:,:,-1])
                        if len(pred.shape) == 2:
                            if pred.shape[1] > len(classnames):
                                pred = pred.reshape(end_idx-start_idx+1, 1, len(classnames), num_azimuths+1)
                            else:
                                pred = pred.reshape(end_idx-start_idx+1, 1, len(classnames), 1)
                        if len(pred.shape) == 4:
                            pred = np.squeeze(pred[:,:,:,-1])
                        if m['do_invert']:
                            gt = 1-gt
                            pred = 1-pred
                        thr = [c['thr_bacc_cl_max'] for c in movrl['id_bacc']]
                        m['id_ge_thr'] = eval_id_ge_thr(gt, pred, thr, classnames=classnames)
                    elif m['pred_type'] == 'loc':
                        gt = gt[:,:,:,:-1] # no squeeze
                        # handle different prediction shapes
                        if len(pred.shape) == 2 and pred.shape[1] > num_azimuths:
                                pred = pred.reshape(end_idx-start_idx+1, 1, len(classnames), num_azimuths+1)
                        if len(pred.shape) == 4:
                            # exclude void bin if present
                            if pred.shape[-1] > num_azimuths:
                                pred = pred[:,:,:,:-1]
                            else:
                                pred = pred[:,:,:,:]
                        thr = [c['thr_bacc_cl_max'] for c in movrl['loc_bacc']]
                        m['loc_ge_thr'] = eval_loc_ge_thr(gt, pred, thr)
                start_idx = end_idx 
                model_stats_cond.append(model_stats)
                print 'model_stats_cond', len(model_stats_cond)
    return model_stats_cond

def summary_pr_auc_id(paths_infer):
    #palette = itrt.cycle(sns.color_palette())
    palette = itrt.cycle(colors_base)
    fig, ax = plt.subplots(1, 1)
    idx = 0
    data = []
    labels = []
    for p in paths_infer:
        m_stats = load_obj(p)
        m_stats_info = load_info(p)
        for m, m_info in zip(m_stats, m_stats_info):
            color_cur = next(palette)
            if m['pred_type'] == 'id':
                pr_auc = np.array([x['pr_auc'] for x in m['id_pr']]).reshape(-1,1)
                data.append(pr_auc[np.logical_not(np.isnan(pr_auc))])
                labels.append(r'%s' % (m_info.replace('conv.', '').rstrip(' ').rstrip(',').rstrip(' '),))
    ax.boxplot(data, 1,
               labels=labels,
              )
    idx += 1
    fig.suptitle('Precision Recall AUC')
    ax.set_title("pooled over all classes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('model')
    ax.set_ylabel('PR AUC')
    ax.set_ylim([0, 1.1])
    ax.grid()
    ax.set_axisbelow(True)
    
def summary_pr_auc_loc(paths_infer):
    #palette = itrt.cycle(sns.color_palette())
    palette = itrt.cycle(colors_base)
    fig, ax = plt.subplots(1, 1)
    idx = 0
    data = []
    labels = []
    for p in paths_infer:
        m_stats = load_obj(p)
        m_stats_info = load_info(p)
        for m, m_info in zip(m_stats, m_stats_info):
            color_cur = next(palette)
            if m['pred_type'] == 'loc':
                bacc = np.array([x['pr_auc'] for x in m['loc_pr']]).reshape(-1,1)
                data.append(bacc[np.logical_not(np.isnan(bacc))])
                labels.append(r'%s' % (m_info.replace('conv.', '').rstrip(' ').rstrip(',').rstrip(' '),))
    ax.boxplot(data, 1,
               labels=labels,
              )
    idx += 1
    fig.suptitle('Precision-Recall AUC')
    ax.set_title("pooled over all location bins")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('model')
    ax.set_ylabel('PR AUC')
    ax.set_ylim([0, 1.1])
    ax.grid()
    ax.set_axisbelow(True)
    
def summary_bacc_id(paths_infer):
    #palette = itrt.cycle(sns.color_palette())
    palette = itrt.cycle(colors_base)
    fig, ax = plt.subplots(1, 1)
    idx = 0
    data = []
    labels = []
    for p in paths_infer:
        m_stats = load_obj(p)
        m_stats_info = load_info(p)
        for m, m_info in zip(m_stats, m_stats_info):
            color_cur = next(palette)
            if m['pred_type'] == 'id':
#                 for x in m['id_bacc']:
#                     print x['num_points'],x['num_pos']
                bacc = np.array([x['bacc_cl_max'] for x in m['id_bacc']]).reshape(-1,1)
                data.append(bacc[np.logical_not(np.isnan(bacc))])
                labels.append(r'%s' % (m_info.replace('conv.', '').replace('.cl. anechoic|', '').replace('brir|', '').rstrip('_').rstrip(' ').rstrip(',').rstrip(' '),))
    ax.boxplot(data, 1,
               labels=labels,
              )
    idx += 1
    fig.suptitle('Balanced Accuracy')
    ax.set_title("pooled over all classes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('model')
    ax.set_ylabel('BAcc')
    ax.set_ylim([0.49, 1.02])
    ax.grid()
    ax.set_axisbelow(True)
    
def summary_bacc_loc(paths_infer):
    #palette = itrt.cycle(sns.color_palette())
    palette = itrt.cycle(colors_base)
    fig, ax = plt.subplots(1, 1)
    idx = 0
    data = []
    labels = []
    for p in paths_infer:
        m_stats = load_obj(p)
        m_stats_info = load_info(p)
        for m, m_info in zip(m_stats, m_stats_info):
            color_cur = next(palette)
            if m['pred_type'] == 'loc':
                bacc = np.array([x['bacc_cl_max'] for x in m['loc_bacc']]).reshape(-1,1)
#                 if data is None:
#                     data = bacc
#                 else:
#                     data = np.concatenate((data, bacc), 1)
                data.append(bacc[np.logical_not(np.isnan(bacc))])
                #labels.append('(%s, %s)' % (os.path.splitext(os.path.basename(p))[0], m['pred_name']))
                labels.append(r'%s' % (m_info.replace('conv.', '').replace('.cl. anechoic|', '').replace('brir|', '').rstrip('_').rstrip(' ').rstrip(',').rstrip(' '),))
#     data = [row[np.logical_not(np.isnan(row))] for row in data.T] # handle nans
    ax.boxplot(data, 1,
               labels=labels,
              )
    idx += 1
    fig.suptitle('Balanced Accuracy')
    ax.set_title("pooled over all location bins")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('model')
    ax.set_ylabel('BAcc')
    ax.set_ylim([0.49, 1.02])
    ax.grid()
    ax.set_axisbelow(True)
    
def summary_bacc_id_nSrcs_box(paths_infer):
    #palette = itrt.cycle(sns.color_palette())
    palette = itrt.cycle(colors_base)
    fig, ax = plt.subplots(1, 1)
    idx = 0
    data = []
    labels = []
    data2 = {}
    name_suffix = 'nSrcs4'
    x = np.load('/home/kashefy/twoears/label_nSrcs_hist_test.npy')
#     print x
    c = x[0]
    print c
    c = c/float(np.sum(c))
    print c
    for p in paths_infer:
        m_stats_nSrcs = load_obj(p+name_suffix)
#         print p, len(m_stats_nSrcs)
#         print m_stats
        m_stats_info = load_info(p)
        for nSrcs_val, m_stats in enumerate(m_stats_nSrcs):
            for m, m_info in zip(m_stats, m_stats_info)[:1]:
                color_cur = next(palette)
                if m['pred_type'] == 'id':
                    for x in m['id_bacc']:
                        print x['num_points'],x['num_pos']
                    bacc = np.array([x['bacc_cl_max'] for x in m['id_bacc']]).flatten()
                    bacc_masked = bacc[np.logical_not(np.isnan(bacc))]
                    data.append(bacc_masked)
                    labels_cur = r'%s' % (m_info.replace('conv.', '').replace('brir|', '')..rstrip('_').rstrip(' ').rstrip(',').rstrip(' '),)
                    print labels_cur, np.median(bacc_masked), bacc_masked.size
                    labels.append('sources:%d%s' % (nSrcs_val, labels_cur))
#                     bacc_masked = bacc_masked*c[nSrcs_val]
                    if labels_cur not in data2:
                        data2[labels_cur] = bacc_masked.tolist()
                    else:
                        data2[labels_cur].extend(bacc_masked.tolist())
                        
    print 'data:',len(data)
    for r,l in zip(data,labels):
        print type(r), l, r.size
    ax.boxplot(data, 1,
               labels=labels,
              )
    idx += 1
    fig.suptitle('Balanced Accuracy')
    ax.set_title("pooled over all classes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('model')
    ax.set_ylabel('BAcc')
    ax.set_ylim([0.49, 1.09])
    ax.grid()
    ax.set_axisbelow(True)
    
    fig2, ax2 = plt.subplots(1, 1)
    print 'data2:',len(data2)    
    for k in data2.keys():
        print type(data2[k]), k, len(data2[k])
    ax2.boxplot(data2.values(), 1,
           labels=data2.keys(),
          )
    idx += 1
    fig2.suptitle('Balanced Accuracy')
    ax2.set_title("pooled over all classes")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax2.set_xlabel('model')
    ax2.set_ylabel('BAcc')
    ax2.set_ylim([0.49, 1.01])
    ax2.grid()
    ax2.set_axisbelow(True)
    
def summary_bacc_id_nSrcs(paths_infer):
    #palette = itrt.cycle(sns.color_palette())    
#     palette = itrt.cycle(colors_base)
    fig, ax = plt.subplots(1, 1)
    idx = 0
    data_mean = {}
    data_std = {}
    name_suffix = 'nSrcs4'
    for p in paths_infer:
        m_stats_nSrcs = load_obj(p+name_suffix)
#         print p, len(m_stats_nSrcs)
#         print m_stats
        m_stats_info = load_info(p)
        for nSrcs_val, m_stats in enumerate(m_stats_nSrcs):
            for m, m_info in zip(m_stats, m_stats_info):
#                 color_cur = next(palette)
                if m['pred_type'] == 'id':
#                     print len(m['id_bacc'])
                    bacc = np.array([x['bacc_cl_max'] for x in m['id_bacc']]).reshape(-1,1)
                    bacc = bacc[np.logical_not(np.isnan(bacc))].T
                    bacc_mean = np.mean(bacc)
                    bacc_std = np.std(bacc)
#                     label_cur = r'%d source(s) %s' % (nSrcs_val, m_info.replace('conv.', '').rstrip(' ').rstrip(',').rstrip(' '),)
                    label_cur = r'%s' % (m_info.replace('conv.', '').replace('brir|', '').rstrip(' ').rstrip(',').rstrip(' '),)
    
#                     print label_cur, nSrcs_val
                    if label_cur not in data_mean:
                        data_mean[label_cur] = []
                        data_std[label_cur] = []
                    data_mean[label_cur].append(bacc_mean)  
                    data_std[label_cur].append(bacc_std)
                    
#     print data_mean.keys()
    width = 0.125
    ind = np.arange(5)
    opacity = 0.7
    error_config = {'ecolor': '0.3'}
    shift = 0
    idx = 0
    for k in data_mean.keys():
        ax.bar(ind+shift,
               data_mean[k],
               width,
               color=colors_base[idx],
               yerr=data_std[k],
               alpha=opacity,
               error_kw=error_config,
               label=k)
        shift += width
        idx+=1
    ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), ncol=1)
    
    fig.suptitle('Balanced Accuracy')
    ax.set_title("pooled over all classes")
    ax.set_xticks(ind + width*2)
    ax.set_xticklabels((ind))
    ax.set_xlabel('number of sources')
    ax.set_ylabel('BAcc')
    ax.set_ylim([0.49, 1.02])
    ax.grid()
    ax.set_axisbelow(True)
    
    x = np.load('/home/kashefy/twoears/label_nSrcs_hist_test.npy')
#     print x
    c = x[0]
    b = x[1]
    ax2 = ax.twinx()
    ax2.plot(b[:-1]+(width*3),c, 'o', color='b', markersize=10, linewidth=10)
    ax2.yaxis.label.set_color('blue')
    ticks = ax2.get_yticks()
    ax2.tick_params(axis='y', colors='blue')
    ax2.set_yticklabels(["%dK" % int(t/1000) for t in ticks])
    ax2.set_ylabel('number of data points')
    
def summary_bacc_loc_nSrcs(paths_infer):
    #palette = itrt.cycle(sns.color_palette())    
#     palette = itrt.cycle(colors_base)
    fig, ax = plt.subplots(1, 1)
    idx = 0
    data_mean = {}
    data_std = {}
    name_suffix = 'nSrcs4'
    for p in paths_infer:
        m_stats_nSrcs = load_obj(p+name_suffix)
#         print p, len(m_stats_nSrcs)
#         print m_stats
        m_stats_info = load_info(p)
        for nSrcs_val, m_stats in enumerate(m_stats_nSrcs):
            for m, m_info in zip(m_stats, m_stats_info):
#                 color_cur = next(palette)
                if m['pred_type'] == 'loc':
#                     print len(m['id_bacc'])
                    bacc = np.array([x['bacc_cl_max'] for x in m['loc_bacc']]).reshape(-1,1)
                    bacc = bacc[np.logical_not(np.isnan(bacc))].T
                    bacc_mean = np.mean(bacc)
                    bacc_std = np.std(bacc)
#                     label_cur = r'%d source(s) %s' % (nSrcs_val, m_info.replace('conv.', '').rstrip(' ').rstrip(',').rstrip(' '),)
                    label_cur = r'%s' % (m_info.replace('conv.', '').replace('brir|', '').rstrip(' ').rstrip(',').rstrip(' '),)
    
#                     print label_cur, nSrcs_val
                    if label_cur not in data_mean:
                        data_mean[label_cur] = []
                        data_std[label_cur] = []
                    data_mean[label_cur].append(bacc_mean)  
                    data_std[label_cur].append(bacc_std)
                    
#     print data_mean.keys()
    width = 0.125
    ind = np.arange(5)
    opacity = 0.7
    error_config = {'ecolor': '0.3'}
    shift = 0
    idx = 0
    for k in data_mean.keys():
        ax.bar(ind+shift,
               data_mean[k],
               width,
               color=colors_base[idx],
               yerr=data_std[k],
               alpha=opacity,
               error_kw=error_config,
               label=k)
        shift += width
        idx+=1
    ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), ncol=1)
    
    fig.suptitle('Balanced Accuracy')
    ax.set_title("pooled over all location bins")
    ax.set_xticks(ind + width*2)
    ax.set_xticklabels((ind))
    ax.set_xlabel('number of sources')
    ax.set_ylabel('BAcc')
    ax.set_ylim([0.49, 1.02])
    ax.grid()
    ax.set_axisbelow(True)
    
    x = np.load('/home/kashefy/twoears/label_nSrcs_hist_test.npy')
#     print x
    c = x[0]
    b = x[1]
    ax2 = ax.twinx()
    ax2.plot(b[:-1]+(width*3),c, 'o', color='b', markersize=10, linewidth=10)
    ax2.yaxis.label.set_color('blue')
    ticks = ax2.get_yticks()
    ax2.tick_params(axis='y', colors='blue')
    ax2.set_yticklabels(["%dK" % int(t/1000) for t in ticks])
    ax2.set_ylabel('number of data points')

def summary_bacc_id_nSrcs_thr(paths_infer):
    #palette = itrt.cycle(sns.color_palette())    
#     palette = itrt.cycle(colors_base)
    fig, ax = plt.subplots(1, 1)
    idx = 0
    data_mean = {}
    data_std = {}
    name_suffix = 'nSrcs4a'
    labels = []
    for p in paths_infer:
        m_stats_nSrcs = load_obj(p+name_suffix)
#         print p, len(m_stats_nSrcs)
#         print m_stats
        m_stats_info = load_info(p)
        for nSrcs_val, m_stats in enumerate(m_stats_nSrcs):
            for m, m_info in zip(m_stats, m_stats_info):
#                 color_cur = next(palette)
                if m['pred_type'] == 'id':
#                     print len(m['id_bacc'])
                    #id_ge_thr:['num_points', 'f1', 'bacc', 'cl_id', 'thr', 'precision', 'classname', 'specificity', 'num_pos', 'sensitivity']
                    bacc = np.array([x['bacc'] for x in m['id_ge_thr']]).reshape(-1,1)
                    bacc = bacc[np.logical_not(np.isnan(bacc))].T
                    bacc_mean = np.mean(bacc)
                    bacc_std = np.std(bacc)
#                     label_cur = r'%d source(s) %s' % (nSrcs_val, m_info.replace('conv.', '').rstrip(' ').rstrip(',').rstrip(' '),)
                    label_cur = r'%s' % (m_info.replace('conv.', '').replace('brir|', '').rstrip('_').rstrip(' ').rstrip(',').rstrip(' '),)
#                     print label_cur, nSrcs_val
                    if label_cur not in data_mean:
                        labels.append(label_cur)
                        data_mean[label_cur] = []
                        data_std[label_cur] = []
                    data_mean[label_cur].append(bacc_mean)  
                    data_std[label_cur].append(bacc_std)
                    
#     print data_mean.keys()
    width = 0.1
    ind = np.arange(5)
    opacity = 0.7
    error_config = {'ecolor': '0.3'}
    shift = 0
    for idx, k in enumerate(labels):
        ax.bar(ind+shift,
               data_mean[k],
               width,
               color=colors_base[idx],
               yerr=data_std[k],
               alpha=opacity,
               error_kw=error_config,
               label=k)
        shift += width
        idx+=1
    ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), ncol=1)
    
    fig.suptitle('Balanced Accuracy')
    ax.set_title("pooled over all classes")
    ax.set_xticks(ind + (width*(int(len(labels)/2))))
    ax.set_xticklabels((ind))
    ax.set_xlabel('number of sources')
    ax.set_ylabel('BAcc')
    ax.set_ylim([0.49, 1.02])
    ax.grid()
    ax.set_axisbelow(True)
    
    x = np.load('/home/kashefy/twoears/label_nSrcs_hist_test.npy')
#     print x
    c = x[0]
    b = x[1]
    ax2 = ax.twinx()
    ax2.plot(b[:-1]+(width*(int(len(labels)/2))),c, 'o', color='b', markersize=10, linewidth=10)
    ax2.yaxis.label.set_color('blue')
    ticks = ax2.get_yticks()
    ax2.tick_params(axis='y', colors='blue')
    ax2.set_yticklabels(["%dK" % int(t/1000) for t in ticks])
    ax2.set_ylabel('number of data points')
    
def summary_bacc_loc_nSrcs_thr(paths_infer):
    #palette = itrt.cycle(sns.color_palette())    
#     palette = itrt.cycle(colors_base)
    for score_key in ['f1', 'bacc', 'precision','sensitivity', 'specificity']:
        fig, ax = plt.subplots(1, 1)
        idx = 0
        data_mean = {}
        data_std = {}
        name_suffix = 'nSrcs4a'
        labels = []
        for p in paths_infer:
            m_stats_nSrcs = load_obj(p+name_suffix)
    #         print p, len(m_stats_nSrcs)
    #         print m_stats
            m_stats_info = load_info(p)
            for nSrcs_val, m_stats in enumerate(m_stats_nSrcs):
                for m, m_info in zip(m_stats, m_stats_info):
    #                 color_cur = next(palette)
                    if m['pred_type'] == 'loc':
    #                     print len(m['id_bacc'])
    #                     bacc = np.array([x['bacc_cl_max'] for x in m['loc_bacc']]).reshape(-1,1)
                        bacc = np.array([x[score_key] for x in m['loc_ge_thr']]).reshape(-1,1)
                        bacc = bacc[np.logical_not(np.isnan(bacc))].T
                        bacc_mean = np.mean(bacc)
                        bacc_std = np.std(bacc)
#                         label_cur = r'%s' % (m_info.replace('conv.', '').replace('brir|', '').rstrip(' ').rstrip(',').rstrip(' '),)
                        label_cur = r'%s' % (m_info.replace('conv.', '').replace('brir|', '').rstrip('_').rstrip(' ').rstrip(',').rstrip(' '),)
    #                     print label_cur, nSrcs_val
                        if label_cur not in data_mean:
                            labels.append(label_cur)
                            data_mean[label_cur] = []
                            data_std[label_cur] = []
                        data_mean[label_cur].append(bacc_mean)  
                        data_std[label_cur].append(bacc_std)

    #     print data_mean.keys()
        width = 0.10
        ind = np.arange(5)
        opacity = 0.7
        error_config = {'ecolor': '0.3'}
        shift = 0
        for idx, k in enumerate(labels):
#             print k, ind, data_mean[k], data_std[k]
            ax.bar(ind+shift,
                   data_mean[k],
                   width,
                   color=colors_base[idx],
                   yerr=data_std[k],
                   alpha=opacity,
                   error_kw=error_config,
                   label=k)
            shift += width
#         ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), ncol=1)

        if score_key == 'bacc':
            fig.suptitle('Balanced Accuracy')
        elif score_key == 'f1':
            fig.suptitle('F1 score')
        else:
            fig.suptitle(score_key)
        ax.set_title("pooled over all location bins")
        ax.set_xticks(ind + (width*(int(len(labels)/2))))
        ax.set_xticklabels((ind))
        ax.set_xlabel('number of sources')
        ax.set_ylabel(score_key)
        if score_key == 'bacc':
            ax.set_ylim([0.49, 1.02])
        ax.grid()
        ax.set_axisbelow(True)

        x = np.load('/home/kashefy/twoears/label_nSrcs_hist_test.npy')
    #     print x
        c = x[0]
        b = x[1]
        ax2 = ax.twinx()
        ax2.plot(b[:-1]+(width*(int(len(labels)/2))), c, 'o', color='b', markersize=10, linewidth=10)
        ax2.yaxis.label.set_color('blue')
        ticks = ax2.get_yticks()
        ax2.tick_params(axis='y', colors='blue')
        ax2.set_yticklabels(["%dK" % int(t/1000) for t in ticks])
        ax2.set_ylabel('number of data points')
