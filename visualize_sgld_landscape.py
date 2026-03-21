import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import os

def get_landscape_data(word, snapshots, word2id):
    if word not in word2id:
        return None
    
    idx = word2id[word]
    word_snapshots = snapshots[:, idx, :]
    
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(word_snapshots)
    return coords_2d

def visualize_word_landscape(word, coords_2d, ax, title, global_bounds, color_map=cm.plasma):

    x = coords_2d[:, 0]
    y = coords_2d[:, 1]
    
    xmin, xmax, ymin, ymax, zmax = global_bounds
    
    res = 120
    X, Y = np.mgrid[xmin:xmax:complex(res), ymin:ymax:complex(res)]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    try:
        kernel = gaussian_kde(coords_2d.T, bw_method=0.6) 
        Z = np.reshape(kernel(positions).T, X.shape)
    except Exception:
        Z = np.zeros_like(X)
    
    ls = LightSource(azdeg=135, altdeg=45)
    
    z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-10)
    rgb = color_map(z_norm)
    shaded_rgb = ls.shade_rgb(rgb, Z, blend_mode='soft')
    
    surf = ax.plot_surface(X, Y, Z, facecolors=shaded_rgb,
                           rstride=1, cstride=1,
                           linewidth=0, antialiased=True, shade=False)
    
    floor_z = -0.05 * Z.max()
    ax.contourf(X, Y, Z, zdir='z', offset=floor_z, cmap=color_map, alpha=0.15)
    
    ax.scatter(x, y, np.full_like(x, floor_z), color='white', s=5, alpha=0.3)
    

    ax.set_title(f" {word} ".upper(), fontsize=18, fontweight='bold', color='#333333', pad=10)

    ax.set_axis_off()
    
    ax.set_zlim(floor_z, zmax * 1.05)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.view_init(elev=30, azim=-60)

def main():
    model_path = 'models/model_sgld.npz'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print("Loading snapshots and generating premium visualizations...")
    data = np.load(model_path, allow_pickle=True)
    word2id = data['word2id'].item()
    snapshots = data['snapshots']
    
    comparison_pairs = [
        ('music', 'violin'),
        ('science', 'thermodynamics')
    ]
    
    os.makedirs('results/sgld_3d_landscapes', exist_ok=True)
    
    plt.rcParams['font.family'] = 'sans-serif'
    
    for gen_word, spec_word in comparison_pairs:
        fig = plt.figure(figsize=(20, 10), facecolor='#FAFAFA')
        
        coords_gen = get_landscape_data(gen_word, snapshots, word2id)
        coords_spec = get_landscape_data(spec_word, snapshots, word2id)
        
        if coords_gen is None or coords_spec is None:
            continue
            
        pad = 0.5
        all_x = np.concatenate([coords_gen[:,0], coords_spec[:,0]])
        all_y = np.concatenate([coords_gen[:,1], coords_spec[:,1]])
        xmin, xmax = all_x.min() - pad, all_x.max() + pad
        ymin, ymax = all_y.min() - pad, all_y.max() + pad
        
        res = 120
        X, Y = np.mgrid[xmin:xmax:complex(res), ymin:ymax:complex(res)]
        positions = np.vstack([X.ravel(), Y.ravel()])
        try:
            k_gen = gaussian_kde(coords_gen.T, bw_method=0.6)
            z_gen = k_gen(positions).max()
        except: z_gen = 0
        try:
            k_spec = gaussian_kde(coords_spec.T, bw_method=0.6)
            z_spec = k_spec(positions).max()
        except: z_spec = 0
        zmax = max(z_gen, z_spec)
        
        global_bounds = (xmin, xmax, ymin, ymax, zmax)
        
        ax1 = fig.add_subplot(121, projection='3d')
        visualize_word_landscape(gen_word, coords_gen, ax1, "General", global_bounds, color_map=cm.plasma)
        
        ax2 = fig.add_subplot(122, projection='3d')
        visualize_word_landscape(spec_word, coords_spec, ax2, "Specific", global_bounds, color_map=cm.plasma)

        fig.text(0.25, 0.85, "Ambiguous / General Meaning", ha='center', fontsize=14, color='gray', style='italic')
        fig.text(0.75, 0.85, "Precise / Specific Meaning", ha='center', fontsize=14, color='gray', style='italic')
        
        output_file = f'results/sgld_3d_landscapes/{gen_word}_vs_{spec_word}.png'
        plt.subplots_adjust(wspace=0.1, hspace=0)
        plt.savefig(output_file, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved: {output_file}")
        plt.close()

    print("\nPremium visualizations complete.")

if __name__ == "__main__":
    main()
