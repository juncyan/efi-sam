import math
import paddle
import paddle.nn.functional as F

def npo2(len):
    """
    Returns the next power of 2 above len
    """
    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """
    len_npo2 = npo2(X.shape[1])
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.shape[1])
    return F.pad(X, pad_tuple, mode='constant', value=0)

class PScan(paddle.autograd.PyLayer):
    @staticmethod
    def pscan(A, X):
        B, D, L, _ = A.shape
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.shape[2]
            Aa = Aa.reshape([B, D, T//2, 2, -1])
            Xa = Xa.reshape([B, D, T//2, 2, -1])

            Xa[:, :, :, 1] = paddle.add(Xa[:, :, :, 1], paddle.multiply(Aa[:, :, :, 1], Xa[:, :, :, 0]))
            Aa[:, :, :, 1] = paddle.multiply(Aa[:, :, :, 0], Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        if Xa.shape[2] == 4:
            Xa[:, :, 1] = paddle.add(Xa[:, :, 1], paddle.multiply(Aa[:, :, 1], Xa[:, :, 0]))
            Aa[:, :, 1] = paddle.multiply(Aa[:, :, 0], Aa[:, :, 1])

            Xa[:, :, 3] = paddle.add(Xa[:, :, 3], paddle.multiply(Aa[:, :, 3], paddle.add(Xa[:, :, 2], paddle.multiply(Aa[:, :, 2], Xa[:, :, 1]))))
        elif Xa.shape[2] == 2:
            Xa[:, :, 1] = paddle.add(Xa[:, :, 1], paddle.multiply(Aa[:, :, 1], Xa[:, :, 0]))
        else:
            return

        Aa = A[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa = X[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa[:, :, 2] = paddle.add(Xa[:, :, 2], paddle.multiply(Aa[:, :, 2], Xa[:, :, 1]))
        Aa[:, :, 2] = paddle.multiply(Aa[:, :, 1], Aa[:, :, 2])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]

            T = Xa.shape[2]
            Aa = Aa.reshape([B, D, T//2, 2, -1])
            Xa = Xa.reshape([B, D, T//2, 2, -1])

            Xa[:, :, 1:, 0] = paddle.add(Xa[:, :, 1:, 0], paddle.multiply(Aa[:, :, 1:, 0], Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0] = paddle.multiply(Aa[:, :, 1:, 0], Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        B, D, L, _ = A.shape
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.shape[2]
            Aa = Aa.reshape([B, D, T//2, 2, -1])
            Xa = Xa.reshape([B, D, T//2, 2, -1])
            
            Xa[:, :, :, 0] = paddle.add(Xa[:, :, :, 0], paddle.multiply(Aa[:, :, :, 0], Xa[:, :, :, 1]))
            Aa[:, :, :, 0] = paddle.multiply(Aa[:, :, :, 1], Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        if Xa.shape[2] == 4:
            Xa[:, :, 2] = paddle.add(Xa[:, :, 2], paddle.multiply(Aa[:, :, 2], Xa[:, :, 3]))
            Aa[:, :, 2] = paddle.multiply(Aa[:, :, 3], Aa[:, :, 2])

            Xa[:, :, 0] = paddle.add(Xa[:, :, 0], paddle.multiply(Aa[:, :, 0], paddle.add(Xa[:, :, 1], paddle.multiply(Aa[:, :, 1], Xa[:, :, 2]))))
        elif Xa.shape[2] == 2:
            Xa[:, :, 0] = paddle.add(Xa[:, :, 0], paddle.multiply(Aa[:, :, 0], Xa[:, :, 1]))
        else:
            return

        Aa = A[:, :, 0:L:2**(num_steps-2)]
        Xa = X[:, :, 0:L:2**(num_steps-2)]
        Xa[:, :, 1] = paddle.add(Xa[:, :, 1], paddle.multiply(Aa[:, :, 1], Xa[:, :, 2]))
        Aa[:, :, 1] = paddle.multiply(Aa[:, :, 2], Aa[:, :, 1])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.shape[2]
            Aa = Aa.reshape([B, D, T//2, 2, -1])
            Xa = Xa.reshape([B, D, T//2, 2, -1])

            Xa[:, :, :-1, 1] = paddle.add(Xa[:, :, :-1, 1], paddle.multiply(Aa[:, :, :-1, 1], Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1] = paddle.multiply(Aa[:, :, 1:, 0], Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        L = X_in.shape[1]
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in)
            X = pad_npo2(X_in)

        A = A.transpose([0, 2, 1])
        X = X.transpose([0, 2, 1])

        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose([0, 2, 1])[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, X = ctx.saved_tensor()
        L = grad_output_in.shape[1]

        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in)
            A_in = pad_npo2(A_in)

        grad_output = grad_output.transpose([0, 2, 1])
        A_in = A_in.transpose([0, 2, 1])
        A = paddle.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1))

        PScan.pscan_rev(A, grad_output)

        Q = paddle.zeros_like(X)
        Q[:, :, 1:] = paddle.add(Q[:, :, 1:], paddle.multiply(X[:, :, :-1], grad_output[:, :, 1:]))

        return Q.transpose([0, 2, 1])[:, :L], grad_output.transpose([0, 2, 1])[:, :L]

pscan = PScan.apply
