library(jsonlite);library(grid);library(svglite)
D<-function(n,s=FALSE)fromJSON(file.path('inputs',n),simplifyVector=s)
cmp<-D('comparisons.json');rev<-D('revision.json');hist<-D('historical.json');pilot<-D('pilot_analysis.json');example<-D('example_group.json');cells<-read.csv('inputs/cells.csv');pr<-lapply(readLines('inputs/pilot_rows.jsonl',warn=FALSE,encoding='UTF-8'),fromJSON)
dir.create('figures_final',showWarnings=FALSE)
C<-list(ink='#233B50',muted='#617786',binary='#456C8B',quality='#9277A4',centered='#078980',orange='#C77348',pale='#EAF1F3',grid='#E3EBEF',light='#BCDCD6')
T<-function(x,y,s,size=9,col=C$ink,bold=FALSE,just='left')grid.text(s,x,y,just=just,gp=gpar(fontfamily='Arial',fontsize=size,col=col,fontface=if(bold)'bold' else 'plain',lineheight=1.12))
L<-function(x1,y1,x2,y2,col=C$muted,lwd=.8,arrow=FALSE,lty=1)grid.lines(c(x1,x2),c(y1,y2),gp=gpar(col=col,lwd=lwd,lty=lty),arrow=if(arrow)grid::arrow(length=unit(1.8,'mm'),type='closed')else NULL)
B<-function(x,y,w,h,fill='#F5F8F9',border='#D9E4E9')grid.roundrect(x,y,w,h,r=unit(1.4,'mm'),gp=gpar(fill=fill,col=border,lwd=.65))
Pt<-function(x,y,col=C$centered,r=1.1,open=FALSE)grid.circle(x,y,r=unit(r,'mm'),gp=gpar(col=col,fill=if(open)'white' else col,lwd=.9))
savefig<-function(name,w,h,fun){cairo_pdf(file.path('figures_final',paste0(name,'.pdf')),width=w,height=h,family='Arial',pointsize=10);fun();dev.off();svglite(file.path('figures_final',paste0(name,'.svg')),width=w,height=h,pointsize=10);fun();dev.off();png(file.path('figures_final',paste0(name,'.png')),width=w,height=h,units='in',res=180,type='cairo',pointsize=10);fun();dev.off()}
method<-function(){grid.newpage();T(.02,.97,'A   From a training question to a policy update',11,bold=TRUE)
ns<-c(.12,.37,.62,.87)
for(j in 1:4)B(ns[j],.805,.21,.17,if(j==4)'#E6F2EF' else '#F5F8F9')
T(.03,.84,'Training input',10,bold=TRUE);T(.03,.785,'Question to policy\n1,500-problem pool',9)
T(.28,.84,'Sample a group',10,bold=TRUE);T(.28,.785,'Current policy\nFour completions',9)
T(.53,.84,'Measure',10,bold=TRUE);T(.53,.785,'Verifier uses reference\nCount generated IDs',8.5)
T(.78,.84,'Reward rule',10,C$centered,TRUE);T(.78,.785,'Binary / quality only\nCentered shaping',9)
for(j in 1:3)L(ns[j]+.107,.805,ns[j+1]-.107,.805,arrow=TRUE)
for(j in 2:4)B(ns[j],.605,.21,.13)
T(.78,.635,'Normalize',10,bold=TRUE);T(.78,.585,'Within the same group',8.6)
T(.53,.635,'Masked loss',10,bold=TRUE);T(.53,.585,'Exclude truncated outputs',8.3)
T(.28,.635,'Update LoRA',10,bold=TRUE);T(.28,.585,'One optimizer step',9)
L(.87,.72,.87,.673,arrow=TRUE);L(.76,.605,.73,.605,arrow=TRUE);L(.51,.605,.48,.605,arrow=TRUE)
L(.37,.673,.37,.715,arrow=TRUE);T(.02,.65,'100 optimizer steps\nin total',8.6,C$muted)
L(.02,.50,.98,.50,col=C$grid)
T(.02,.465,'B   One recorded group',10.5,bold=TRUE);T(.02,.425,'Qwen2.5 / seed 42 / step 36',8.5,C$muted)
xs<-c(.025,.14,.275,.41,.54);hd<-c('Output','Tokens','Correct','Reward','Advantage');for(j in 1:5)T(xs[j],.365,hd[j],8.2,C$muted)
for(i in 1:4){y<-.31-(i-1)*.065;vals<-c(paste0('#',i),length(example$completion_ids[[i]]),if(example$passed[[i]])'yes' else 'no',sprintf('%.3f',example$rewards[[i]]),sprintf('%+.3f',example$normalized_advantage[[i]]));for(j in 1:5)T(xs[j],y,vals[j],9,if(j==3)if(example$passed[[i]])C$centered else C$orange else C$ink)}
T(.02,.045,'Selected k = 3 example; none truncated.',8.5,C$muted)
L(.67,.075,.67,.47,col=C$grid);T(.70,.465,'C   Independent evaluation',10.2,bold=TRUE)
B(.84,.365,.275,.095,'#EDF5F3');T(.84,.365,'Final adapter',9.5,bold=TRUE,just='center')
L(.37,.54,.37,.515);L(.37,.515,.99,.515);L(.99,.515,.99,.40);L(.99,.40,.97,.40,arrow=TRUE)
T(.70,.265,'300 MATH + 300 GSM8K\nHeld-out questions; greedy decoding',8.3);L(.84,.316,.84,.293,arrow=TRUE)
L(.84,.224,.84,.197,arrow=TRUE);B(.84,.13,.275,.125);T(.84,.13,'Strict / tolerant scoring\nPaired outcome comparisons',8.8,just='center')}
savefig('fig01_method',6.4,4.9,method)
mechanism<-function(){grid.newpage();T(.02,.96,'A   When do the two centers survive group normalization?',10.5,bold=TRUE);xx<-seq(.25,.81,length.out=5)
B(xx[4],.505,.13,.79,'#EDF6F3',border=NA)
for(k in 0:4){x<-xx[k+1];T(x,.875,paste0('k = ',k),9.8,bold=TRUE,just='center');for(j in 1:4)Pt(x+(j-2.5)*.018,.82,if(j<=k)C$centered else '#C4D0D7',r=.95,open=j>k);T(x,.755,c('Off','Off','Same center','May differ','Common shift\ncancels')[k+1],8.4,if(k==3)C$centered else C$muted,just='center')}
T(.02,.66,'B   Observed group counts',10.5,bold=TRUE);T(.89,.66,'k = 3 by seed',8.7,bold=TRUE,just='center');T(.89,.62,'42 / 123 / 2026',8,C$muted,just='center')
g<-Filter(function(x)grepl('centered_budget_s',x$run),rev$groups);pats<-c('Qwen2.5','Qwen3','minicpm5_external');labs<-c('Qwen2.5','Qwen3','MiniCPM')
for(i in 1:3){z<-Filter(function(x)grepl(pats[i],x$run,fixed=TRUE),g);z<-z[order(sapply(z,function(x)match(as.integer(sub('.*_s','',x$run)),c(42,123,2026))))];stopifnot(length(z)==3,all(sapply(z,function(x)x$n)==200));counts<-sapply(0:4,function(k)sum(sapply(z,function(x)x$k[[as.character(k)]])));y<-.535-(i-1)*.165;T(.02,y+.025,labs[i],9.5,bold=TRUE);T(.02,y-.022,'600 groups',8.4,C$muted);for(j in 1:5){Pt(xx[j],y,if(j==4)C$centered else C$binary,r=sqrt(counts[j]/600)*4.1,open=j!=4);T(xx[j],y-.055,paste0(counts[j],'/600'),8.4,just='center')};kk<-sapply(z,function(x)x$k[['3']]);T(.89,y,paste(kk,collapse=' / '),9,just='center');T(.89,y-.045,'each / 200',8,C$muted,just='center')}
T(.02,.045,'Necessary opportunity, not a causal explanation of final accuracy.',9,C$muted)}
savefig('fig02_mechanism',6.4,4.2,mechanism)
primary<-function(){grid.newpage();T(.02,.97,'Does shaping yield better answers at the fixed endpoint?',11,bold=TRUE);T(.02,.925,'A   Accuracy (%): every paired seed',9.5,bold=TRUE);T(.55,.925,'B   Difference (pp): centered - comparator',8.7,bold=TRUE)
xa<-function(v).18+(v-30)/45*.29;xe<-function(v).56+(v+18)/24*.30
for(v in c(30,45,60,75))T(xa(v),.875,v,8,C$muted,just='center');for(v in c(-15,-10,-5,0,5))T(xe(v),.875,v,8,C$muted,just='center');T(.94,.875,'Holm p',8,C$muted,just='center')
ys<-c(.79,.67,.55,.43,.265,.145)
for(i in 1:6){a<-cmp[[i]];y<-ys[i];col<-if(a$control=='Binary')C$binary else C$quality;T(.02,y+.043,paste(a$label,'vs',a$control),9,bold=TRUE)
for(j in 1:3){yy<-y-(j-2)*.022;T(.12,yy,c('42','123','2026')[j],7.8,C$muted,just='right');b<-a$base[[j]]/3;t<-a$treated[[j]]/3;L(xa(b),yy,xa(t),yy,col='#A6B8C3',lwd=1);Pt(xa(b),yy,col,.8,TRUE);Pt(xa(t),yy,C$centered,.6)}
L(xe(0),y-.045,xe(0),y+.045,col='#B5C3CC',lty=2);ci<-unlist(a$conditional_item_ci95)*100;cr<-unlist(a$crossed_seed_item_ci95)*100;L(xe(cr[1]),y,xe(cr[2]),y,col=C$light,lwd=5);L(xe(ci[1]),y,xe(ci[2]),y,col=C$centered,lwd=1.5);Pt(xe(a$difference*100),y,C$centered,1.05);T(.94,y,if(a$p<.0001)formatC(a$p,format='e',digits=2) else sprintf('%.3f',a$p),8,just='center');T(.70,y-.043,sprintf('Mean %+.2f pp',a$difference*100),8,C$muted,just='center')}
L(.02,.345,.98,.345,col=C$grid);T(.02,.332,'External family: separate two-test correction',8.4,C$muted)
T(.02,.055,'Open: comparator   Filled: centered',8.5,C$muted);T(.55,.055,'Dark: item CI   Pale: seed-item CI',8.5,C$muted);T(.02,.022,'300 questions per dataset; 3 seeds; intervals are 95%.',8.4,C$muted)}
savefig('fig03_primary',6.4,5.8,primary)
scoring<-function(){grid.newpage();T(.02,.96,'Does relaxing presentation rules change the comparison?',11,bold=TRUE);T(.02,.91,'A   Each matched seed: strict and tolerant effect',9.5,bold=TRUE);T(.82,.91,'Mean difference (pp)',9,bold=TRUE,just='center');T(.82,.87,'Strict / tolerant',8.5,C$muted,just='center');xx<-function(v).31+(v+16)/22*.30
for(v in c(-15,-10,-5,0,5))T(xx(v),.865,v,8,C$muted,just='center')
for(i in 1:6){a<-cmp[[i]];y<-.79-(i-1)*.093;T(.02,y,paste(a$label,a$control,sep='\n'),8.5);ss<-unlist(a$seed_differences)*100;tt<-(unlist(a$treated_tolerant)-unlist(a$base_tolerant))/3;for(j in 1:3){yy<-y+(j-2)*.019;L(xx(ss[j]),yy,xx(tt[j]),yy,col='#B4C4CD');Pt(xx(ss[j]),yy,C$binary,.9,TRUE);Pt(xx(tt[j]),yy,C$orange,.55)};T(.82,y,sprintf('%+.2f / %+.2f',mean(ss),mean(tt)),9,just='center')}
L(.02,.23,.98,.23,col=C$grid);T(.02,.205,'B   Qwen2.5 / GSM8K: full-cap parser sensitivity',9.5,bold=TRUE)
q<-read.csv('inputs/gsm_scoring.csv');T(.02,.16,'Reward',8.3,C$muted);T(.24,.16,'Strict counts / 300',8.3,C$muted);T(.50,.16,'Tolerant counts / 300',8.3,C$muted);T(.80,.16,'Means (%)',8.3,C$muted)
for(i in 1:3){z<-q[q$arm==c('Binary','Quality only','Centered')[i],];y<-.12-(i-1)*.045;T(.02,y,z$arm[1],8.5);T(.24,y,paste(z$strict,collapse=' / '),8.5);T(.50,y,paste(z$tolerant,collapse=' / '),8.5);T(.80,y,sprintf('%.2f / %.2f',mean(z$strict)/3,mean(z$tolerant)/3),8.5)}
}
savefig('fig04_scoring',6.4,6.0,scoring)
style<-function(){par(family='Arial',bty='n',las=1,mgp=c(2,.5,0),tcl=-.2,col.axis=C$ink,col.lab=C$ink,fg=C$ink,cex=1)}
budget<-function(){layout(matrix(1:4,2,byrow=TRUE));style();models<-c('Qwen2.5-1.5B-Instruct','Qwen3-4B-Instruct');arms<-c('binary','grs_only','centered_budget');cl<-c(C$binary,C$quality,C$centered);pchs<-c(1,2,15)
for(mi in 1:2)for(di in 1:2){ds<-c('math','gsm')[di];z<-subset(cells,model==models[mi]&dataset==ds&prompt_style=='compact'&metric=='strict_passed'&arm%in%arms&seed%in%c(42,123,2026));stopifnot(nrow(z)==27);par(mar=c(4.2,4.2,3.1,1),cex.axis=.85,cex.lab=.9);xl<-range(z$mean_tokens);xl<-xl+c(-1,1)*max(diff(xl)*.13,8);yl<-range(z$accuracy*100);yl<-yl+c(-1,1)*max(diff(yl)*.12,1);plot(NA,xlim=xl,ylim=yl,xlab='Mean output tokens / question',ylab='Strict accuracy (%)');abline(h=axTicks(2),v=axTicks(1),col=C$grid,lwd=.6)
for(a in 1:3)for(k in 1:3){v<-subset(z,arm==arms[a]&cap==c(256,512,1024)[k]);points(v$mean_tokens,v$accuracy*100,pch=pchs[k],col=cl[a],cex=.85)};keys<-paste(format(z$mean_tokens,digits=16),format(z$accuracy,digits=16),sep='/');for(kk in unique(keys)){ii<-which(keys==kk);if(length(ii)>1&&!(mi==2&&di==2))text(z$mean_tokens[ii[1]],z$accuracy[ii[1]]*100+diff(yl)*.055,paste0('n=',length(ii)),cex=.78,col=C$ink)};if(mi==2&&di==2){stopifnot(all(table(keys)==3));text(xl[1]+diff(xl)*.05,yl[2]-diff(yl)*.08,'Every location: n = 3',adj=0,cex=.78)};mtext(paste0('27 records / ',length(unique(keys)),' locations'),3,line=.1,adj=0,cex=.78);mtext(paste0(letters[(mi-1)*2+di],'   ',c('Qwen2.5','Qwen3')[mi],' / ',c('MATH','GSM8K')[di]),3,line=1.2,adj=0,font=2,cex=.95);if(mi==1&&di==1)legend('topleft',c('Binary','Quality only','Centered'),col=cl,pch=16,bty='n',cex=.78);if(mi==1&&di==2)legend('bottomright',c('256 prefix','512 prefix','1,024 output'),pch=pchs,bty='n',cex=.78)} }
savefig('fig05_budget',6.4,5.2,budget)
ha<-c('e0','e2','e3','e5','e6');hl<-c('Base','Binary','Length penalty','Tiny-GAR','Budget-GAR');hc<-c(C$muted,C$binary,C$quality,C$orange,C$centered);seeds<-c(42,123,2026);nm<-function(a,s)if(a=='e0')'e0_math500_256'else paste0('mathhard_',a,'_s',s)
historical<-function(){layout(matrix(1:4,2,byrow=TRUE));style();par(oma=c(2.6,0,0,0));rates<-sapply(ha,function(a)sapply(seeds,function(s)hist$summary[[nm(a,s)]]$accuracy*100))
par(mar=c(4,7,2.8,.5),cex.axis=.8,cex.lab=.85);plot(NA,xlim=c(54,64),ylim=c(.5,5.5),yaxt='n',xlab='Historical accuracy (%)',ylab='');abline(v=c(56,60,64),col=C$grid);axis(2,5:1,hl,tick=FALSE);for(i in 1:5){js<-if(i==1)1 else 1:3;for(j in js)points(rates[j,i],6-i+if(i==1)0 else (j-2)*.16,pch=if(i==1)18 else c(16,17,15)[j],col=hc[i],cex=.85)};mtext('a   Every evaluated seed',3,line=.9,adj=0,font=2,cex=.9)
par(mar=c(4,2,2.8,1),cex.axis=.8,cex.lab=.85);plot(NA,xlim=c(-5,5),ylim=c(.5,5.5),yaxt='n',xlab='Difference from Base (pp)',ylab='');abline(v=0,lty=2,col=C$muted);for(i in 2:5){e<-Filter(function(x)x$a==ha[i]&&x$b=='e0',hist$conditional_effects)[[1]];ci<-unlist(e$ci95)*100;segments(ci[1],6-i,ci[2],6-i,col=hc[i],lwd=1.4);points(e$delta*100,6-i,pch=18,col=hc[i],cex=.9)};mtext('b   Conditional 95% intervals',3,line=.9,adj=0,font=2,cex=.9)
par(mar=c(4.4,4.6,2.8,.5),cex.axis=.8,cex.lab=.85);plot(NA,xlim=c(15,85),ylim=c(55,64),xlab='Nonzero reward SD (%)',ylab='Historical accuracy (%)');abline(h=rates[1,1],lty=2,col=C$muted);text(80,rates[1,1]+.5,'Base',adj=1,cex=.78,col=C$muted);for(i in 2:5)for(j in 1:3){k<-nm(ha[i],seeds[j]);points((1-hist$training[[k]]$metric_means$frac_reward_zero_std)*100,rates[j,i],pch=c(16,17,15)[j],col=hc[i],cex=.8)};mtext('c   Signal is not task gain',3,line=.9,adj=0,font=2,cex=.9)
par(mar=c(4.4,4.7,2.8,1),cex.axis=.8,cex.lab=.85);plot(NA,xlim=c(145,165),ylim=c(130,185),xlab='Successful tokens / success',ylab='Failed tokens / success');for(i in 1:5){js<-if(i==1)1 else 1:3;for(j in js){z<-hist$summary[[nm(ha[i],seeds[j])]];points(z$mean_reencoded_success,z$total_reencoded_per_success-z$mean_reencoded_success,pch=if(i==1)18 else c(16,17,15)[j],col=hc[i],cex=.85)}};mtext('d   Both outputs incur cost',3,line=.9,adj=0,font=2,cex=.9);legend('bottomleft',paste('Seed',seeds),pch=c(16,17,15),bty='n',cex=.78);par(fig=c(0,1,0,1),mar=c(0,0,0,0),oma=c(0,0,0,0),new=TRUE);plot.new();legend('bottom',hl,col=hc,pch=16,horiz=TRUE,bty='n',cex=.75)}
savefig('fig06_historical',6.4,6.0,historical)
pilotdraw<-function(){layout(matrix(c(1,2,3,4),2,byrow=TRUE));style();models<-c('Qwen2.5-1.5B-Instruct','Qwen3-4B-Instruct');cc<-c('budget_compact','budget_unconstrained','prompt_256','prompt_512');ll<-c('512 vs 256\ncompact','512 vs 256\nfree','Compact vs free\n256','Compact vs free\n512')
for(mi in 1:2){par(mar=c(3.5,6.7,3.2,.8),cex.axis=.8,cex.lab=.85);plot(NA,xlim=c(-50,100),ylim=c(.5,4.7),yaxt='n',xlab='Paired accuracy difference (pp)',ylab='');axis(2,4:1,ll,tick=FALSE,cex.axis=.78);abline(v=0,lty=2,col=C$muted);for(k in 1:4)for(si in 1:2){z<-Filter(function(x)x$model==models[mi]&&x$contrast==cc[k]&&x$score==c('strict','passed_legacy_verifier')[si],pilot$contrasts)[[1]];ci<-unlist(z$paired_item_percentile95)*100;y<-5-k+c(.12,-.12)[si];cl<-c(C$binary,C$orange)[si];segments(ci[1],y,ci[2],y,col=cl,lwd=1);points(z$difference*100,y,pch=c(16,1)[si],col=cl,cex=.8)};mtext(paste0(letters[mi],'   ',c('Qwen2.5','Qwen3')[mi],' protocol effects'),3,line=1,adj=0,font=2,cex=.9);legend('topright',c('Strict','Last number'),pch=c(16,1),col=c(C$binary,C$orange),bty='n',cex=.78)}
for(mi in 1:2){par(mar=c(4,4.6,3.4,1),cex.axis=.8,cex.lab=.85);plot(NA,xlim=c(0,275),ylim=c(0,535),xlab='Tokens at cap 256',ylab='Tokens at cap 512');abline(a=0,b=1,col=C$grid,lty=2);abline(v=256,h=512,col=C$grid,lty=3);rr<-Filter(function(x)x$model==models[mi],pr)
for(st in c('compact','unconstrained')){z<-Filter(function(x)x$prompt_style==st,rr);tasks<-unique(sapply(z,function(x)x$task_id));xy<-t(sapply(tasks,function(t){p<-Filter(function(x)x$task_id==t,z);p<-p[order(sapply(p,function(x)x$max_new_tokens))];c(p[[1]]$generated_tokens,p[[2]]$generated_tokens,as.numeric(p[[2]]$stop_reason=='length'))}));keys<-apply(xy,1,paste,collapse='/');for(k in unique(keys)){j<-which(keys==k);pt<-xy[j[1],];cl<-if(st=='compact')C$binary else C$orange;points(pt[1],pt[2],pch=if(pt[3])17 else 1,col=cl,cex=.7*sqrt(length(j)));if(length(j)>1)text(pt[1]-9,pt[2]+if(pt[3]==0)35 else if(st=='compact')-35 else -65,paste0(if(st=='compact')'C: n=' else 'F: n=',length(j)),cex=.78,col=cl,adj=1)}};mtext(paste0(letters[mi+2],'   Same-question output pairs'),3,line=1.6,adj=0,font=2,cex=.9);mtext('Distinct paired-length patterns; area counts repeats',3,line=.3,adj=0,cex=.78);legend('topleft',c('Compact','Free','EOS at 512','Cap 512 reached'),col=c(C$binary,C$orange,C$muted,C$muted),pch=c(16,16,1,17),bty='n',cex=.78)} }
savefig('fig07_pilot',6.4,6,pilotdraw)
writeLines(capture.output(sessionInfo()),'figures_final/R_session.txt')
tradeoff<-function(){grid.newpage();d<-D('minicpm_tradeoff.json',TRUE);T(.02,.96,'MiniCPM / GSM8K: shorter outputs come with lower accuracy',10.8,bold=TRUE)
T(.02,.88,'A   Paired training seeds',9.3,bold=TRUE)
xx<-function(v).075+(v-80)/115*.40;yy<-function(v).23+(v-48)/24*.53
for(v in seq(50,70,5)){L(.075,yy(v),.475,yy(v),C$grid);T(.06,yy(v),as.character(v),8.3,C$muted,just='right')}
for(v in c(90,120,150,180)){L(xx(v),.23,xx(v),.76,C$grid);T(xx(v),.20,as.character(v),8.3,just='center')}
L(.075,.23,.475,.23);T(.275,.12,'Mean generated tokens / question',8.5,just='center');T(.075,.80,'Strict accuracy (%)',8.5)
for(seed in c(42,123,2026)){a<-d[d$run==paste0('binary_s',seed),];b<-d[d$run==paste0('centered_budget_s',seed),];L(xx(a$tokens/300),yy(a$correct/3),xx(b$tokens/300),yy(b$correct/3),C$muted,.8);Pt(xx(a$tokens/300),yy(a$correct/3),C$binary,1.25,TRUE);Pt(xx(b$tokens/300),yy(b$correct/3),C$centered,1.25);T(xx(a$tokens/300)+.008,yy(a$correct/3)+.015,as.character(seed),8.3,C$binary)}
a<-d[d$run=='base',];grid.points(unit(xx(a$tokens/300),'npc'),unit(yy(a$correct/3),'npc'),pch=18,size=unit(2.3,'mm'),gp=gpar(col=C$orange));T(xx(a$tokens/300)+.012,yy(a$correct/3)-.02,'Base',8.2,C$orange)
L(.515,.11,.515,.87,C$grid);T(.55,.88,'B   Accuracy and output',10,bold=TRUE)
T(.55,.79,'Three seeds pooled; 900 evaluations / arm',8.2,C$muted);T(.77,.72,'Binary',9,C$binary,TRUE,just='center');T(.93,.72,'Centered',9,C$centered,TRUE,just='center')
labels<-c('Strict accuracy','Tokens / question','Tokens / success');summ<-function(z)c(100*sum(z$correct)/sum(z$n),sum(z$tokens)/sum(z$n),sum(z$tokens)/sum(z$correct)); a<-summ(d[startsWith(d$run,'binary_s'),]); b<-summ(d[startsWith(d$run,'centered_budget_s'),]); fmt<-function(x)c(sprintf('%.2f%%',x[1]),sprintf('%.2f',x[2:3]));v1<-fmt(a);v2<-fmt(b)
for(i in 1:3){y<-.64-(i-1)*.12;T(.55,y,labels[i],8.1);T(.77,y,v1[i],9.1,just='center');T(.93,y,v2[i],9.1,just='center')}
grid.rect(.76,.245,width=.46,height=.165,gp=gpar(fill=C$pale,col=NA));L(.535,.162,.535,.328,C$centered,2);T(.55,.28,sprintf('%+.2f pp strict accuracy',b[1]-a[1]),10,C$ink,TRUE);T(.55,.21,sprintf('%+.2f%% tokens / question',100*(b[2]/a[2]-1)),9,C$centered,TRUE);T(.55,.125,'300 unique questions; failures included.\nOutput only; no latency or total-cost claim.',8.1,C$muted)
T(.02,.035,'Open blue: binary   Filled teal: centered   Diamond: one untrained reference',8.2,C$muted)}
savefig('fig08_tradeoff',6.4,4.5,tradeoff)
