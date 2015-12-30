# Make plot showing effect of data augmentation on geoquery
pdf('fig-geo-augment.pdf')
op = par(family='serif')
x = c(100, 200, 300, 400, 500)
y_orig = c(37, 58, 69, 74, 76)
y_aug = c(44, 65, 75, 83, 87)
y_aug = c(44, 64, 74, 83, 87)
plot(x, y_orig, ylim=c(35, 90), col='red', 
     xlab='Number of original training examples', ylab='Accuracy (%)',
     cex=1.5, cex.axis=1.5, cex.lab=1.5)
lines(x, y_orig, col='red')
points(x, y_aug, col='blue', cex=1.5)
lines(x, y_aug, col='blue')

legend('topleft', c('No augmented data', 'With augmented data'),
       col= c('red', 'blue'), lty=1, cex=1.5)
dev.off()
par(op)
