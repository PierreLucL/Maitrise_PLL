"""Figures et bilan descriptif du couple 100/300, sans réentraîner."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args();root=args.folder
    d=json.loads((root/'comparison.json').read_text())
    m=np.load(root/'maps_and_learning.npz')
    rows=d['comparisons'];names=[n.replace('Rég. ','') for n in d['region_names']]
    n=len(names);counts=d['sizes'];ratios=np.array(counts)/np.mean(counts)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    ### Même échelle ET même ordre pour les deux durées : pas de tri opportuniste.
    limit=float(np.quantile(np.concatenate([np.abs(m[k]).ravel() for k in m.files if k.startswith('current_')]),.99))
    cmap=LinearSegmentedColormap.from_list('activity',['#101010','#ffffff','#b21818'])
    colors=[plt.get_cmap('tab10')(i) for i in range(n)]
    plot_meta=dict(current_limit=limit,current_percentile=99,window=[60,120],
                   order_reference='Pic de l’activité RNN à 100 passages dans 60–120 s, ordre fixé pour les deux modèles',
                   current_saturation={},activity_outside_0_1={})
    for di,recon in enumerate(d['reconstruction']):
        fig,axes=plt.subplots(n,n+1,figsize=(15,13),gridspec_kw={'height_ratios':ratios},layout='constrained')
        for ti in range(n):
            order=d['orders'][ti]
            im_a=axes[ti,0].imshow(m[f'activity_{di}_{ti}'][order],aspect='auto',cmap=cmap,vmin=0,vmax=1,extent=[60,120,counts[ti],0],interpolation='nearest')
            axes[ti,0].set_ylabel(f'{names[ti]}\n{counts[ti]} unités',color=colors[ti])
            for si in range(n):
                im_c=axes[ti,si+1].imshow(m[f'current_{di}_{ti}_{si}'][order],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,extent=[60,120,counts[ti],0],interpolation='nearest')
            for col,ax in enumerate(axes[ti]):
                ax.set_yticks([]);ax.set_xticks([60,90,120] if ti==n-1 else [])
                if ti==0:ax.set_title('Activité\ndu modèle' if col==0 else 'Depuis\n'+names[col-1],fontsize=11)
                if ti==n-1:ax.set_xlabel('Temps (s)')
                for spine in ax.spines.values():spine.set_color(colors[ti] if col==0 else colors[col-1]);spine.set_linewidth(.8)
        fig.colorbar(im_a,ax=axes[:,0],location='bottom',shrink=.8,pad=.025,label='Activité du modèle')
        fig.colorbar(im_c,ax=axes[:,1:],location='bottom',shrink=.55,pad=.025,label='Courant signé · échelle identique à 100 et 300',extend='neither')
        fig.suptitle(f"Souris 410 · graine 2026 · {recon['nRunTrain']} passages\nUnités et échelles communes · fenêtre 60–120 s · saturation au P99 commun",fontsize=13)
        fig.savefig(root/f"maps_{recon['nRunTrain']}.png",dpi=180);plt.close(fig)
        plot_meta['current_saturation'][str(recon['nRunTrain'])]=float(np.mean(np.concatenate([np.abs(m[f'current_{di}_{ti}_{si}']).ravel()>limit for ti in range(n) for si in range(n)])))
        plot_meta['activity_outside_0_1'][str(recon['nRunTrain'])]=float(np.mean(np.concatenate([((m[f'activity_{di}_{ti}']<0)|(m[f'activity_{di}_{ti}']>1)).ravel() for ti in range(n)])))
    (root/'plot_metadata.json').write_text(json.dumps(plot_meta,indent=2,ensure_ascii=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
    plots=[('pca_norm_r','Corrélation des normes PCA10',-1,1,'RdBu_r'),
           ('pca_norm_ratio','Amplitude PCA10 : ratio 300 / 100',None,None,'viridis'),
           ('cca','Première corrélation canonique',0,1,'viridis'),
           ('cca_mean','Moyenne des 10 corrélations canoniques',0,1,'viridis')]
    for ax,(key,title,vmin,vmax,cmap_name) in zip(axes.ravel(),plots):
        values=np.array([r['cca'][0] if key=='cca' else np.mean(r['cca']) if key=='cca_mean' else r[key] for r in rows]).reshape(n,n)
        im=ax.imshow(values,cmap=cmap_name,vmin=vmin,vmax=vmax)
        ax.set_xticks(range(n),names,rotation=35,ha='right');ax.set_yticks(range(n),names)
        ax.set_xlabel('Source');ax.set_ylabel('Cible');ax.set_title(title)
        for i in range(n):
            for j in range(n):
                value=values[i,j];rgba=im.cmap(im.norm(value));light=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='black' if light>.55 else 'white')
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('410 · graine 2026 · 100 → 300 passages\nMême session et initialisation · PCA–CCA descriptive sur toute la session',fontsize=14)
    fig.savefig(root/'comparaison_pca_cca.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for di,(key,color) in enumerate([('pvar_a','#1f77b4'),('pvar_b','#d95f02')]):
        nt=d['reconstruction'][di]['nRunTrain'];p=m[key]
        ax.plot(np.arange(1,nt+1),p[:nt],color=color,lw=1.3,label=f'{nt} passages : apprentissage',alpha=.8)
        ax.plot(np.arange(nt+1,len(p)+1),p[nt:],color=color,ls='--',label=f'{nt} passages : évaluation sans mise à jour')
    ax.set(xlabel='Passage sur la même session',ylabel='pVar',title='Courbes d’apprentissage enregistrées · aucun nouvel entraînement')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(root/'apprentissage.png',dpi=170);plt.close(fig)
    rec=d['reconstruction'];pv=[r['pvar'] for r in rec]
    norm=np.array([r['pca_norm_r'] for r in rows]);amp=np.array([r['pca_norm_ratio'] for r in rows])
    cc=np.array([r['cca'] for r in rows]);ct=d['controls']
    lines=['# Souris 410, graine 2026 — comparaison appariée 100 / 300', '',
           'Analyse du 13 septembre 2026. Le modèle à 300 passages a terminé en 56 h 06 min 51 s sur Narval (2796272_0).', '',
           '## Intégrité et appariement', '',
           f"Empreinte SHA-256 du PKL 300 vérifiée contre Narval : `{d['sha256'][1]}`.", '',
           'Contrôles réussis : '+', '.join(d['checks'])+'.', '',
           f"Écart maximal de pVar sur les {rec[0]['nRunTrain']} premiers passages communs : {d['training_prefix_pvar_max_abs_difference']:.3g}. Les deux entraînements repartent des mêmes conditions ; il ne s’agit pas d’une reprise depuis checkpoint. La version historique tt est conservée.", '',
           '## Reconstruction', '',
           '| Mesure | 100 passages | 300 passages |','|---|---:|---:|',
           f'| pVar recalculée sur la session | {pv[0]:.6f} | {pv[1]:.6f} |',
           f"| Temps total enregistré dans le PKL | {rec[0]['runtime_sec']/3600:.2f} h | {rec[1]['runtime_sec']/3600:.2f} h |", '',
           f"Gain de pVar : {(pv[1]-pv[0])*100:.2f} points ; baisse de l’erreur quadratique de {100*(1-(1-pv[1])/(1-pv[0])):.1f} % sur cette cible commune. Ce gain ne constitue pas une validation sur de nouvelles données.", '',
           '| Région | pVar 100 | pVar 300 | Corrélation dérivées 100 | Corrélation dérivées 300 |','|---|---:|---:|---:|---:|']
    for ra,rb in zip(rec[0]['regions'],rec[1]['regions']):lines.append(f"| {ra['region']} | {ra['pvar']:.4f} | {rb['pvar']:.4f} | {ra['derivative_r']:.4f} | {rb['derivative_r']:.4f} |")
    lines+=['','La corrélation des dérivées utilise les différences temporelles de chaque unité, puis une corrélation sur les valeurs regroupées de la région ; ce n’est pas la médiane des corrélations par unité.','',
            '## Courants par unité et dynamiques dominantes','',
            f'- Corrélation des normes PCA10 : médiane **{np.median(norm):.3f}**, étendue {norm.min():.3f}–{norm.max():.3f}.',
            f'- Première CCA : médiane **{np.median(cc[:,0]):.3f}** ; moyenne des dix CCA : médiane **{np.median(cc.mean(1)):.3f}**.',
            f'- Ratio moyen des normes PCA10 (300/100) : médiane **{np.median(amp):.3f}**, étendue **{amp.min():.3f}–{amp.max():.3f}**.',
            f"- Corrélation des sommes signées, diagnostic complémentaire : médiane {np.median([r['signed_r'] for r in rows]):.3f} ; {sum(r['signed_r']<0 for r in rows)}/36 négatives.",
            f"- Médiane des 36 corrélations médianes par unité sans alignement : {np.median([r['unit_r_median'] for r in rows]):.3f}.", '',
            'Chaque contribution est calculée par J[cible,source] @ RNN[source], sans somme préalable des unités cibles. La PCA est centrée temporellement par unité et conserve dix composantes non blanchies. Les CCA décrivent des sous-espaces et ne garantissent pas signe ou amplitude identiques. Ces conventions reprennent les analyses historiques ; leur identité exacte avec les scripts des auteurs reste à vérifier.', '',
            '## Transfert temporel et correspondance des sources', '',
            f"PCA et CCA apprises avant 230 s, puis évaluées à partir de 250 s. CCA test médiane : **{np.median([r['own']['test_cca'] for r in ct]):.3f}**. La bonne source dépasse les cinq autres en CCA dans **{sum(r['cca_margin']>0 for r in ct)}/36** cas ; marge médiane {np.median([r['cca_margin'] for r in ct]):+.4f}. Selon la norme : {sum(r['norm_margin']>0 for r in ct)}/36 cas.", '',
            'Ces contrôles comparent 100 à 300 pour une seule graine. Ils ne mesurent pas une amélioration de la spécificité entre graines à 300 et ne sont pas directement équivalents aux 108 comparaisons historiques. Les contrôles par décalage n’ont pas été répétés ici. Le réseau a appris toute la session ; seul l’alignement est évalué sur un autre bloc.', '',
            '## Figures', '',
            '- comparaison_pca_cca.png : les 36 contributions, sans sélection de région.',
            '- maps_100.png et maps_300.png : fenêtre fixée à 60–120 s, activité du modèle à gauche, sources en colonnes, mêmes unités et ordre fixés depuis le modèle 100, hauteurs proportionnelles aux effectifs.',
            f'- Échelle commune des courants : ±{limit:.5g}, P99 absolu regroupé sur les deux modèles. Les extrêmes sont saturés à l’affichage ; aucune normalisation par panneau. Fractions de saturation dans plot_metadata.json.',
            '- apprentissage.png : les passages libres sont séparés des passages entraînés.', '',
            '## Portée pour la maîtrise', '',
            'Ce test isole la durée à données, graine et version communes, sous les contrôles numériques rapportés. Il permet de décrire ce qui change entre 100 et 300, pas de conclure que 300 rend les courants plus reproductibles entre graines ou entre sessions. Il faudra retenir des caractéristiques dont la variabilité technique est suffisamment faible pour les comparaisons longitudinales ; aucune différence d’âge n’est testée ici.', '',
            'Les 36 contributions sont dépendantes et ne sont pas 36 souris. Une CCA1 élevée, une somme stable ou une pVar améliorée ne constitue pas une preuve causale ou anatomique.', '',
            'Détails et provenance : comparison.json. Cartes de la fenêtre et courbes : maps_and_learning.npz. Les PKL complets permettent de régénérer les cartes sur les autres temps. Scripts : scripts/analysis/compare_training_durations.py et report_training_durations.py.', '']
    (root/'bilan.md').write_text('\n'.join(lines))
    print(root/'bilan.md')


if __name__=='__main__':main()
